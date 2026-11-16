using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

/// <summary>
/// MXFP4 (OCP micro-scaling FP4, GGML type 39). A block is 32 values in 17
/// bytes: one E8M0 exponent, then sixteen bytes of E2M1 nibbles. Nibble
/// <c>j</c> holds element <c>j</c> in its low half and element <c>j + 16</c> in
/// its high half — the same half-split the 32-value Q4 formats use, which is why
/// the kernels below are shaped like the Q4_NL ones. What differs is the scale:
/// one byte of power-of-two exponent instead of an fp16 multiply pair.
///
/// There is no encoder for this format: the nibbles have to come from a
/// quantizer that searched the E2M1 grid (ggml's <c>quantize_row_mxfp4_ref</c>
/// does it with a 16-entry table per value), and SharpMind only decodes.
/// </summary>
public static partial class QuantizationKernels
{
    private const int MXFP4_BlockBytes = 17;

    /// <summary>
    /// The E2M1 values a nibble indexes: <c>{0, .5, 1, 1.5, 2, 3, 4, 6}</c> for
    /// the low half of the range and their negations for the high half. Every
    /// value is exactly representable in float, so the decode has no rounding.
    /// </summary>
    private static readonly float[] kvalues_mxfp4 =
        [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f];

    private static readonly Vector256<float> MXFP4Lo = Vector256.Create(
        kvalues_mxfp4[0], kvalues_mxfp4[1], kvalues_mxfp4[2], kvalues_mxfp4[3],
        kvalues_mxfp4[4], kvalues_mxfp4[5], kvalues_mxfp4[6], kvalues_mxfp4[7]);
    private static readonly Vector256<float> MXFP4Hi = Vector256.Create(
        kvalues_mxfp4[8], kvalues_mxfp4[9], kvalues_mxfp4[10], kvalues_mxfp4[11],
        kvalues_mxfp4[12], kvalues_mxfp4[13], kvalues_mxfp4[14], kvalues_mxfp4[15]);

    /// <summary>
    /// Eight decoded weights from one nibble byte group: low nibbles for the
    /// first sixteen elements of the block, high nibbles for the second sixteen.
    /// Two eight-lane permutes plus a blend stand in for the gather AVX2 lacks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> MXFP4Values8(byte* qs, int i)
    {
        var idx = Avx2.ConvertToVector256Int32(qs + (i & 15));
        idx = (i & 16) != 0
            ? Avx2.ShiftRightLogical(idx, 4)
            : Avx2.And(idx, Vector256.Create(0x0F));
        var lo = Avx2.PermuteVar8x32(MXFP4Lo, idx);
        var hi = Avx2.PermuteVar8x32(MXFP4Hi, idx);
        var ge8 = Avx2.CompareGreaterThan(idx, Vector256.Create(7)).AsSingle();
        return Avx.BlendVariable(lo, hi, ge8);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotMXFP4_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * MXFP4_BlockBytes + b * MXFP4_BlockBytes;
            float d = E8M0ToFloat_Scalar(block[0]);
            byte* qs = block + 1;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * (d * kvalues_mxfp4[nib]);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotMXFP4_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        var vacc2 = Vector256<float>.Zero;
        var vacc3 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * MXFP4_BlockBytes + b * MXFP4_BlockBytes;
            float d = E8M0ToFloat_Scalar(block[0]);
            byte* qs = block + 1;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            int i = 0;
            // Four independent chains per 32-value block, accumulated across the
            // whole row and reduced once (see VecDotQ4_NL_AVX2).
            for (; i <= blockEnd - 32; i += 32)
            {
                var w0 = MXFP4Values8(qs, i);
                var w1 = MXFP4Values8(qs, i + 8);
                var w2 = MXFP4Values8(qs, i + 16);
                var w3 = MXFP4Values8(qs, i + 24);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), Avx.Multiply(w0, vd)));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 8]), Avx.Multiply(w1, vd)));
                vacc2 = Avx.Add(vacc2, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 16]), Avx.Multiply(w2, vd)));
                vacc3 = Avx.Add(vacc3, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 24]), Avx.Multiply(w3, vd)));
            }
            for (; i <= blockEnd - 16; i += 16)
            {
                var w0 = MXFP4Values8(qs, i);
                var w1 = MXFP4Values8(qs, i + 8);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), Avx.Multiply(w0, vd)));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 8]), Avx.Multiply(w1, vd)));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var w = MXFP4Values8(qs, i);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), Avx.Multiply(w, vd)));
            }
            for (; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * (d * kvalues_mxfp4[nib]);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(Avx.Add(vacc0, vacc1), Avx.Add(vacc2, vacc3)));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotMXFP4_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * MXFP4_BlockBytes + b * MXFP4_BlockBytes;
            float d = E8M0ToFloat_Scalar(block[0]);
            byte* qs = block + 1;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                var w0 = Avx.Multiply(MXFP4Values8(qs, i), vd);
                var w1 = Avx.Multiply(MXFP4Values8(qs, i + 8), vd);
                vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), w0, vacc0);
                vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i + 8]), w1, vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var w = Avx.Multiply(MXFP4Values8(qs, i), vd);
                vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), w, vacc0);
            }
            for (; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * (d * kvalues_mxfp4[nib]);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulMXFP4_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotMXFP4_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulMXFP4_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotMXFP4_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotMXFP4_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulMXFP4_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotMXFP4_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulMXFP4_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotMXFP4_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotMXFP4_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulMXFP4_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotMXFP4_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulMXFP4_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotMXFP4_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotMXFP4_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static void ReadMXFP4_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[MXFP4_BlockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = E8M0ToFloat_Scalar(buf[0]);
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int half = qk / 2;
                int nib = (j < half) ? (buf[1 + j] & 0x0F) : (buf[1 + j - half] >> 4);
                data[blockStart + j] = d * kvalues_mxfp4[nib];
            }
        }
    }
}
