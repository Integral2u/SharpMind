using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    // Eight consecutive 2-bit codes for block-relative positions idx..idx+7.
    // Inside a 32-value group the packed bytes are contiguous and the plane
    // (idx/128) and shift ((idx%128)/32) are constant, so one widened load and
    // one shift replace the per-element divide/modulo that used to fill a
    // stack buffer before every vector multiply.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> Q2KCodes8(byte* qs, int idx)
    {
        var w = Avx2.ConvertToVector256Int32(qs + ((idx >> 7) << 5) + (idx & 31));
        w = Avx2.And(Avx2.ShiftRightLogical(w, (byte)((((idx >> 5) & 3) << 1))), Vector256.Create(3));
        return Avx.ConvertToVector256Single(w);
    }

    // Same eight codes, with the byte address and plane shift already resolved by
    // the caller. Both are constant across a 16-value window inside a 32-value
    // group, and deriving them from the running index inside the 8-lane loop
    // cost 14 of that loop's 24 instructions.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> Q2KCodesAt(byte* q, int shift)
    {
        var w = Avx2.And(Avx2.ShiftRightLogical(Avx2.ConvertToVector256Int32(q), (byte)shift), Vector256.Create(3));
        return Avx.ConvertToVector256Single(w);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ2K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 16;
            float dSuper = HalfToFloat_Scalar(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 82));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 8 + j * 2;
                    float s0 = scales[isc] & 0x0F;
                    float m0 = scales[isc] >> 4;
                    for (int l = 0; l < 16 && basePos + l < blockEnd; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += input[b * QK_K + idx - colBlockStart] * (s0 * v * dSuper - m0 * minSuper);
                    }

                    float s1 = scales[isc + 1] & 0x0F;
                    float m1 = scales[isc + 1] >> 4;
                    for (int l = 0; l < 16 && basePos + 16 + l < blockEnd; l++)
                    {
                        int idx = basePos + 16 + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += input[b * QK_K + idx - colBlockStart] * (s1 * v * dSuper - m1 * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ2K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 16;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 82));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 8 + j * 2;
                    float s0 = scales[isc] & 0x0F;
                    float m0 = scales[isc] >> 4;
                    var vs0 = Vector256.Create(s0 * dSuper);
                    var vm0 = Vector256.Create(m0 * minSuper);

                    int subRem = Math.Min(16, blockEnd - basePos);
                    bool vec8 = (basePos & 7) == 0;
                    int l = 0;
                    for (; vec8 && l <= subRem - 8; l += 8)
                    {
                        var vv = Q2KCodes8(qs, basePos + l);
                        var vi = Vector256.LoadUnsafe(ref pIn[basePos + l]);
                        var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0), vm0));
                        sum += MathHelpers.HSum256_Avx(res);
                    }
                    for (; l < subRem; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += pIn[idx] * (s0 * v * dSuper - m0 * minSuper);
                    }

                    float s1 = scales[isc + 1] & 0x0F;
                    float m1 = scales[isc + 1] >> 4;
                    var vs1 = Vector256.Create(s1 * dSuper);
                    var vm1 = Vector256.Create(m1 * minSuper);

                    int bPos1 = basePos + 16;
                    int subRem2 = Math.Min(16, blockEnd - bPos1);
                    bool vec8b = (bPos1 & 7) == 0;
                    int l2 = 0;
                    for (; vec8b && l2 <= subRem2 - 8; l2 += 8)
                    {
                        var vv = Q2KCodes8(qs, bPos1 + l2);
                        var vi = Vector256.LoadUnsafe(ref pIn[bPos1 + l2]);
                        var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1), vm1));
                        sum += MathHelpers.HSum256_Avx(res);
                    }
                    for (; l2 < subRem2; l2++)
                    {
                        int idx = bPos1 + l2;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += pIn[idx] * (s1 * v * dSuper - m1 * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ2K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        var vacc2 = Vector256<float>.Zero;
        var vacc3 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 16;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 82));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 8 + j * 2;
                    float s0 = scales[isc] & 0x0F;
                    float m0 = scales[isc] >> 4;
                    float s1 = scales[isc + 1] & 0x0F;
                    float m1 = scales[isc + 1] >> 4;
                    var vs0 = Vector256.Create(s0 * dSuper);
                    var vm0 = Vector256.Create(m0 * minSuper);
                    var vs1 = Vector256.Create(s1 * dSuper);
                    var vm1 = Vector256.Create(m1 * minSuper);

                    int subRem = Math.Min(16, blockEnd - basePos);
                    int bPos1 = basePos + 16;
                    int subRem2 = Math.Min(16, blockEnd - bPos1);
                    // A whole 32-value group maps to 32 contiguous packed bytes at
                    // one shift, so it decodes as four independent chains. The
                    // unpack→shift→convert→mul→sub→fma chain is ~21 cycles and the
                    // four chains overlap; a single accumulator runs them end to
                    // end instead, which is where the per-group cost was going.
                    if ((basePos & 31) == 0 && subRem == 16 && subRem2 == 16)
                    {
                        byte* q = qs + ((basePos >> 7) << 5) + (basePos & 31);
                        int sh = (((basePos >> 5) & 3) << 1);
                        var w0 = Q2KCodesAt(q, sh);
                        var w1 = Q2KCodesAt(q + 8, sh);
                        var w2 = Q2KCodesAt(q + 16, sh);
                        var w3 = Q2KCodesAt(q + 24, sh);
                        // Materialise all four "vw" values before any FMA so their
                        // live ranges overlap: otherwise the scheduler shares one
                        // temp register across the four mul/sub pairs and the
                        // chains serialise on it again.
                        var u0 = Avx.Subtract(Avx.Multiply(w0, vs0), vm0);
                        var u1 = Avx.Subtract(Avx.Multiply(w1, vs0), vm0);
                        var u2 = Avx.Subtract(Avx.Multiply(w2, vs1), vm1);
                        var u3 = Avx.Subtract(Avx.Multiply(w3, vs1), vm1);
                        vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[basePos]), u0, vacc0);
                        vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[basePos + 8]), u1, vacc1);
                        vacc2 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[basePos + 16]), u2, vacc2);
                        vacc3 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[basePos + 24]), u3, vacc3);
                        continue;
                    }

                    // The widened load below only matches the packed layout when
                    // the 8-value window sits inside one 32-value group, which an
                    // 8-aligned start guarantees. Unaligned column offsets fall
                    // back to the scalar tail.
                    bool vec8 = (basePos & 7) == 0 && (basePos & 31) + subRem <= 32;
                    byte* q0 = qs + ((basePos >> 7) << 5) + (basePos & 31);
                    int sh0 = (((basePos >> 5) & 3) << 1);
                    int l = 0;
                    for (; vec8 && l <= subRem - 8; l += 8)
                    {
                        var vv = Q2KCodesAt(q0, sh0);
                        q0 += 8;
                        var vi = Vector256.LoadUnsafe(ref pIn[basePos + l]);
                        var vw = Avx.Subtract(Avx.Multiply(vv, vs0), vm0);
                        vacc0 = Fma.MultiplyAdd(vi, vw, vacc0);
                    }
                    for (; l < subRem; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += pIn[idx] * (s0 * v * dSuper - m0 * minSuper);
                    }

                    bool vec8b = (bPos1 & 7) == 0 && (bPos1 & 31) + subRem2 <= 32;
                    byte* q1 = qs + ((bPos1 >> 7) << 5) + (bPos1 & 31);
                    int sh1 = (((bPos1 >> 5) & 3) << 1);
                    int l2 = 0;
                    for (; vec8b && l2 <= subRem2 - 8; l2 += 8)
                    {
                        var vv = Q2KCodesAt(q1, sh1);
                        q1 += 8;
                        var vi = Vector256.LoadUnsafe(ref pIn[bPos1 + l2]);
                        var vw = Avx.Subtract(Avx.Multiply(vv, vs1), vm1);
                        vacc1 = Fma.MultiplyAdd(vi, vw, vacc1);
                    }
                    for (; l2 < subRem2; l2++)
                    {
                        int idx = bPos1 + l2;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += pIn[idx] * (s1 * v * dSuper - m1 * minSuper);
                    }
                }
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(Avx.Add(vacc0, vacc1), Avx.Add(vacc2, vacc3)));
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulQ2K_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ2K_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ2K_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ2K_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ2K_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static void ReadQ2K_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 84;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);

            float dSuper = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[80]));
            float minSuper = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[82]));

            for (int i = 0; i < QK_K && blockStart + i < n; i++)
            {
                int pairIdx = i / 16;
                byte pair = buf[pairIdx];
                float s = pair & 0x0F;
                float m = pair >> 4;
                int qsByte = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int quant = (buf[16 + qsByte] >> qsShift) & 3;
                data[blockStart + i] = (s * quant * dSuper) - (m * minSuper);
            }
        }
    }

    public static unsafe void QuantizedMatMulQ2K_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ2K_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ2K_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ2K_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ2K_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ2K_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ2K_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ2K_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ2K_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ2K_FMA(pInRow, rawWeights, col, K);
            });
        }
    }
}
