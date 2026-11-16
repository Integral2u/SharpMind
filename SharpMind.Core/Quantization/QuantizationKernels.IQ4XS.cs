using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

/// <summary>
/// IQ4_XS (GGML type 23). A 256-value super-block in 136 bytes: an fp16
/// super-scale, a 16-bit high half of the eight 6-bit sub-block scales, four
/// bytes of their low halves, then 128 bytes of nibble pairs. Sub-block
/// <c>ib</c> covers 32 values as 16 bytes — low nibbles for the first sixteen,
/// high for the second — and indexes the same 16-entry <c>iq4nl</c> table the
/// 32-value IQ4_NL block does; what differs is the 6-bit scale (biased by 32)
/// and the 256-value super-block.
///
/// Decode only: encoding needs the grid search <c>quantize_iq4_xs</c> does.
/// </summary>
public static partial class QuantizationKernels
{
    private const int IQ4XS_BlockBytes = 136;
    private const int IQ4XS_SubBlockValues = 32;

    /// <summary>
    /// One of the eight 6-bit sub-block scales: the low four bits live in
    /// <c>scales_l[ib/2]</c> (high nibble for odd <c>ib</c>), the top two in
    /// <c>scales_h</c> at <c>2*ib</c>. The value is biased by 32, so it is
    /// signed once the bias is subtracted.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe int IQ4XSScale(int ib, ushort scalesH, byte* scalesL) =>
        ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scalesH >> (2 * ib)) & 0x03) << 4);

    public static void ReadIQ4_XS_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[IQ4XS_BlockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            ushort scalesH = Unsafe.ReadUnaligned<ushort>(ref buf[2]);
            int valid = Math.Min(qk, n - blockStart);

            for (int ib = 0; ib < qk / IQ4XS_SubBlockValues; ib++)
            {
                int basePos = ib * IQ4XS_SubBlockValues;
                if (basePos >= valid) break;

                int ls = ((buf[4 + ib / 2] >> (4 * (ib % 2))) & 0x0F) | (((scalesH >> (2 * ib)) & 0x03) << 4);
                float dl = d * (ls - 32);
                int subEnd = Math.Min(IQ4XS_SubBlockValues, valid - basePos);

                for (int j = 0; j < subEnd; j++)
                {
                    int byteIdx = 8 + ib * 16 + (j < 16 ? j : j - 16);
                    int nib = j < 16 ? (buf[byteIdx] & 0x0F) : (buf[byteIdx] >> 4);
                    data[blockStart + basePos + j] = dl * kvalues_iq4nl[nib];
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ4_XS_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * IQ4XS_BlockBytes + b * IQ4XS_BlockBytes;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            ushort scalesH = *(ushort*)(block + 2);
            byte* scalesL = block + 4;
            byte* qs = block + 8;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            for (int ib = 0; ib < QK_K / IQ4XS_SubBlockValues; ib++)
            {
                int basePos = ib * IQ4XS_SubBlockValues;
                if (basePos >= blockEnd) break;

                float dl = d * (IQ4XSScale(ib, scalesH, scalesL) - 32);
                byte* qsSub = qs + ib * 16;
                int subEnd = Math.Min(IQ4XS_SubBlockValues, blockEnd - basePos);
                for (int i = 0; i < subEnd; i++)
                {
                    int nib = (i < 16) ? (qsSub[i] & 0x0F) : (qsSub[i - 16] >> 4);
                    sum += pIn[basePos + i] * (dl * kvalues_iq4nl[nib]);
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ4_XS_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        var vacc2 = Vector256<float>.Zero;
        var vacc3 = Vector256<float>.Zero;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * IQ4XS_BlockBytes + b * IQ4XS_BlockBytes;
            float d = HalfToFloat_F16C(*(ushort*)block);
            ushort scalesH = *(ushort*)(block + 2);
            byte* scalesL = block + 4;
            byte* qs = block + 8;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            for (int ib = 0; ib < QK_K / IQ4XS_SubBlockValues; ib++)
            {
                int basePos = ib * IQ4XS_SubBlockValues;
                if (basePos >= blockEnd) break;

                float dl = d * (IQ4XSScale(ib, scalesH, scalesL) - 32);
                byte* qsSub = qs + ib * 16;
                int subEnd = Math.Min(IQ4XS_SubBlockValues, blockEnd - basePos);
                float* pSub = pIn + basePos;
                var vd = Vector256.Create(dl);
                int i = 0;
                // Four independent chains, one per 8-value group of the 32-value
                // sub-block; reduced once at the end of the row.
                for (; i <= subEnd - 32; i += 32)
                {
                    vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pSub[i]),
                        Avx.Multiply(IQ4NLValues8(qsSub, i), vd)));
                    vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pSub[i + 8]),
                        Avx.Multiply(IQ4NLValues8(qsSub, i + 8), vd)));
                    vacc2 = Avx.Add(vacc2, Avx.Multiply(Vector256.LoadUnsafe(ref pSub[i + 16]),
                        Avx.Multiply(IQ4NLValues8(qsSub, i + 16), vd)));
                    vacc3 = Avx.Add(vacc3, Avx.Multiply(Vector256.LoadUnsafe(ref pSub[i + 24]),
                        Avx.Multiply(IQ4NLValues8(qsSub, i + 24), vd)));
                }
                for (; i <= subEnd - 8; i += 8)
                {
                    vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pSub[i]),
                        Avx.Multiply(IQ4NLValues8(qsSub, i), vd)));
                }
                for (; i < subEnd; i++)
                {
                    int nib = (i < 16) ? (qsSub[i] & 0x0F) : (qsSub[i - 16] >> 4);
                    sum += pSub[i] * (dl * kvalues_iq4nl[nib]);
                }
            }
        }

        sum += MathHelpers.HSum256_Avx(Avx.Add(Avx.Add(vacc0, vacc1), Avx.Add(vacc2, vacc3)));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotIQ4_XS_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * IQ4XS_BlockBytes + b * IQ4XS_BlockBytes;
            float d = HalfToFloat_F16C(*(ushort*)block);
            ushort scalesH = *(ushort*)(block + 2);
            byte* scalesL = block + 4;
            byte* qs = block + 8;
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            for (int ib = 0; ib < QK_K / IQ4XS_SubBlockValues; ib++)
            {
                int basePos = ib * IQ4XS_SubBlockValues;
                if (basePos >= blockEnd) break;

                float dl = d * (IQ4XSScale(ib, scalesH, scalesL) - 32);
                byte* qsSub = qs + ib * 16;
                int subEnd = Math.Min(IQ4XS_SubBlockValues, blockEnd - basePos);
                float* pSub = pIn + basePos;
                var vd = Vector256.Create(dl);
                int i = 0;
                for (; i <= subEnd - 16; i += 16)
                {
                    vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pSub[i]),
                        Avx.Multiply(IQ4NLValues8(qsSub, i), vd), vacc0);
                    vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pSub[i + 8]),
                        Avx.Multiply(IQ4NLValues8(qsSub, i + 8), vd), vacc1);
                }
                for (; i <= subEnd - 8; i += 8)
                {
                    vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pSub[i]),
                        Avx.Multiply(IQ4NLValues8(qsSub, i), vd), vacc0);
                }
                for (; i < subEnd; i++)
                {
                    int nib = (i < 16) ? (qsSub[i] & 0x0F) : (qsSub[i - 16] >> 4);
                    sum += pSub[i] * (dl * kvalues_iq4nl[nib]);
                }
            }
        }

        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulIQ4_XS_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotIQ4_XS_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulIQ4_XS_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotIQ4_XS_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotIQ4_XS_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulIQ4_XS_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotIQ4_XS_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulIQ4_XS_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotIQ4_XS_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotIQ4_XS_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulIQ4_XS_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotIQ4_XS_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulIQ4_XS_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotIQ4_XS_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotIQ4_XS_FMA(pInRow, rawWeights, col, K);
            });
        }
    }
}
