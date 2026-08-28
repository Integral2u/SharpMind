using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ6K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d = HalfToFloat_Scalar(*(ushort*)(block + 208));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int nOff = curBlockStart; nOff < blockEnd; nOff += 128)
            {
                byte* pql = ql + (nOff == 0 ? 0 : 64);
                byte* pqh = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                int halfRem = Math.Min(128, blockEnd - nOff);
                for (int l = 0; l < 32 && l < halfRem; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = b * QK_K + nOff + l - colBlockStart;
                    int i2 = b * QK_K + nOff + l + 32 - colBlockStart;

                    if (i2 >= b * QK_K + blockEnd - colBlockStart)
                    {
                        if (i1 < b * QK_K + blockEnd - colBlockStart)
                            sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                        break;
                    }

                    int i3 = b * QK_K + nOff + l + 64 - colBlockStart;
                    int i4 = b * QK_K + nOff + l + 96 - colBlockStart;

                    sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                    sum += input[i2] * (d * psc[is_ + 2] * (q2v - 32));
                    sum += input[i3] * (d * psc[is_ + 4] * (q3v - 32));
                    sum += input[i4] * (d * psc[is_ + 6] * (q4v - 32));
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ6K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
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
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d = HalfToFloat_F16C(*(ushort*)(block + 208));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int nOff = curBlockStart; nOff < blockEnd; nOff += 128)
            {
                byte* pql = ql + (nOff == 0 ? 0 : 64);
                byte* pqh = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                int halfRem = Math.Min(128, blockEnd - nOff);

                int l = 0;
                for (; l + 103 < halfRem && l < 32; l += 8)
                {
                    int is_ = l / 16;
                    Q6KCodes8(pql, pqh, l, out var q1, out var q2, out var q3, out var q4);
                    float s1 = d * psc[is_ + 0], s2 = d * psc[is_ + 2], s3 = d * psc[is_ + 4], s4 = d * psc[is_ + 6];
                    var vw1 = Avx.Subtract(Avx.Multiply(q1, Vector256.Create(s1)), Vector256.Create(32 * s1));
                    var vw2 = Avx.Subtract(Avx.Multiply(q2, Vector256.Create(s2)), Vector256.Create(32 * s2));
                    var vw3 = Avx.Subtract(Avx.Multiply(q3, Vector256.Create(s3)), Vector256.Create(32 * s3));
                    var vw4 = Avx.Subtract(Avx.Multiply(q4, Vector256.Create(s4)), Vector256.Create(32 * s4));
                    vacc0 = Avx.Add(Avx.Multiply(Vector256.LoadUnsafe(ref pIn[nOff + l]), vw1), vacc0);
                    vacc1 = Avx.Add(Avx.Multiply(Vector256.LoadUnsafe(ref pIn[nOff + l + 32]), vw2), vacc1);
                    vacc2 = Avx.Add(Avx.Multiply(Vector256.LoadUnsafe(ref pIn[nOff + l + 64]), vw3), vacc2);
                    vacc3 = Avx.Add(Avx.Multiply(Vector256.LoadUnsafe(ref pIn[nOff + l + 96]), vw4), vacc3);
                }

                for (; l < halfRem && l < 32; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = nOff + l;
                    int i2 = nOff + l + 32;

                    if (i2 >= blockEnd)
                    {
                        if (i1 < blockEnd)
                            sum += pIn[i1] * (d * psc[is_ + 0] * (q1v - 32));
                        break;
                    }

                    int i3 = nOff + l + 64;
                    int i4 = nOff + l + 96;

                    float v1 = d * psc[is_ + 0] * (q1v - 32);
                    float v2 = d * psc[is_ + 2] * (q2v - 32);
                    float v3 = d * psc[is_ + 4] * (q3v - 32);
                    float v4 = d * psc[is_ + 6] * (q4v - 32);

                    sum += pIn[i1] * v1;
                    sum += pIn[i2] * v2;
                    sum += pIn[i3] * v3;
                    sum += pIn[i4] * v4;
                }
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(Avx.Add(vacc0, vacc1), Avx.Add(vacc2, vacc3)));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ6K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
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
            byte* ql = block;
            byte* qh = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d = HalfToFloat_F16C(*(ushort*)(block + 208));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int nOff = curBlockStart; nOff < blockEnd; nOff += 128)
            {
                byte* pql = ql + (nOff == 0 ? 0 : 64);
                byte* pqh = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                int halfRem = Math.Min(128, blockEnd - nOff);

                int l = 0;
                for (; l + 103 < halfRem && l < 32; l += 8)
                {
                    int is_ = l / 16;
                    Q6KCodes8(pql, pqh, l, out var q1, out var q2, out var q3, out var q4);
                    float s1 = d * psc[is_ + 0], s2 = d * psc[is_ + 2], s3 = d * psc[is_ + 4], s4 = d * psc[is_ + 6];
                    var vw1 = Fma.MultiplySubtract(q1, Vector256.Create(s1), Vector256.Create(32 * s1));
                    var vw2 = Fma.MultiplySubtract(q2, Vector256.Create(s2), Vector256.Create(32 * s2));
                    var vw3 = Fma.MultiplySubtract(q3, Vector256.Create(s3), Vector256.Create(32 * s3));
                    var vw4 = Fma.MultiplySubtract(q4, Vector256.Create(s4), Vector256.Create(32 * s4));
                    vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[nOff + l]), vw1, vacc0);
                    vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[nOff + l + 32]), vw2, vacc1);
                    vacc2 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[nOff + l + 64]), vw3, vacc2);
                    vacc3 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[nOff + l + 96]), vw4, vacc3);
                }

                for (; l < halfRem && l < 32; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = nOff + l;
                    int i2 = nOff + l + 32;

                    if (i2 >= blockEnd)
                    {
                        if (i1 < blockEnd)
                            sum += pIn[i1] * (d * psc[is_ + 0] * (q1v - 32));
                        break;
                    }

                    int i3 = nOff + l + 64;
                    int i4 = nOff + l + 96;

                    float v1 = d * psc[is_ + 0] * (q1v - 32);
                    float v2 = d * psc[is_ + 2] * (q2v - 32);
                    float v3 = d * psc[is_ + 4] * (q3v - 32);
                    float v4 = d * psc[is_ + 6] * (q4v - 32);

                    sum += pIn[i1] * v1;
                    sum += pIn[i2] * v2;
                    sum += pIn[i3] * v3;
                    sum += pIn[i4] * v4;
                }
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(Avx.Add(vacc0, vacc1), Avx.Add(vacc2, vacc3)));
        return (float)sum;
    }
    
    public static unsafe void ReadQ6K_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 210;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            int valid = Math.Min(QK_K, n - blockStart);
            reader.Read(buf);

            fixed (byte* pBuf = buf)
            {
                byte* ql = pBuf;
                byte* qh = ql + 128;
                sbyte* scales = (sbyte*)(qh + 64);
                float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(pBuf + 128 + 64 + 16));

                for (int nOff = 0; nOff < valid; nOff += 128)
                {
                    int qlOff = nOff == 0 ? 0 : 64;
                    int qhOff = nOff == 0 ? 0 : 32;
                    int scOff = nOff == 0 ? 0 : 8;

                    int halfRem = Math.Min(128, valid - nOff);
                    for (int l = 0; l < 32 && l < halfRem; l++)
                    {
                        int is_ = l / 16;
                        int q1 = (ql[qlOff + l] & 0x0F) | ((qh[qhOff + l] & 0x03) << 4);
                        int q2 = (ql[qlOff + l + 32] & 0x0F) | (((qh[qhOff + l] >> 2) & 0x03) << 4);
                        int q3 = ((ql[qlOff + l] >> 4) & 0x0F) | (((qh[qhOff + l] >> 4) & 0x03) << 4);
                        int q4 = ((ql[qlOff + l + 32] >> 4) & 0x0F) | (((qh[qhOff + l] >> 6) & 0x03) << 4);

                        int idx1 = nOff + l;
                        int idx2 = nOff + l + 32;

                        if (idx2 >= valid)
                        {
                            if (idx1 < valid)
                                data[blockStart + idx1] = d * scales[scOff + is_ + 0] * (q1 - 32);
                            break;
                        }

                        int idx3 = nOff + l + 64;
                        int idx4 = nOff + l + 96;

                        data[blockStart + idx1] = d * scales[scOff + is_ + 0] * (q1 - 32);
                        data[blockStart + idx2] = d * scales[scOff + is_ + 2] * (q2 - 32);
                        data[blockStart + idx3] = d * scales[scOff + is_ + 4] * (q3 - 32);
                        data[blockStart + idx4] = d * scales[scOff + is_ + 6] * (q4 - 32);
                    }
                }
            }
        }
    }
    public static unsafe void QuantizedMatMulQ6K_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ6K_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ6K_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ6K_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ6K_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ6K_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ6K_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ6K_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ6K_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ6K_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ6K_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            QuantizedMatMul_Serial_Wrapper(VecDotQ6K_FMA, input, rawWeights, output, M, K, N);
            return;
        }
        // M > 1 (prefill/training): dequantize each column once, then run the
        // blocked four-row microkernel. See QuantBlockedColumns.
        QuantBlockedMatMul(DequantColumnQ6K, input, rawWeights, output, M, K, N, parallel: false);
    }

    public static unsafe void QuantizedMatMulQ6K_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ6K_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            QuantBlockedMatMul(DequantColumnQ6K, input, rawWeights, output, M, K, N, parallel: true);
        }
    }

}
