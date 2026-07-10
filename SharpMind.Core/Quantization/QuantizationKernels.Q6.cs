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
                byte* pql = ql + nOff / 2;
                byte* pqh = qh + nOff / 4;
                sbyte* psc = scales + nOff / 16;
                for (int l = 0; l < 32; l++)
                {
                    int is_ = l / 16;

                    int qlNib1 = (pql[l / 2] >> (4 * (l & 1))) & 0x0F;
                    int qhBits1 = (pqh[l / 4] >> (2 * (l & 3))) & 3;
                    int q1 = qlNib1 | (qhBits1 << 4);
                    int i1 = nOff + l;
                    if (i1 < blockEnd)
                        sum += input[b * QK_K + i1 - colBlockStart] * (d * psc[is_ + 0] * (q1 - 32));

                    int qlNib2 = (pql[l / 2 + 16] >> (4 * (l & 1))) & 0x0F;
                    int qhBits2 = (pqh[l / 4 + 8] >> (2 * (l & 3))) & 3;
                    int q2 = qlNib2 | (qhBits2 << 4);
                    int i2 = nOff + l + 32;
                    if (i2 < blockEnd)
                        sum += input[b * QK_K + i2 - colBlockStart] * (d * psc[is_ + 2] * (q2 - 32));

                    int qlNib3 = (pql[l / 2 + 32] >> (4 * (l & 1))) & 0x0F;
                    int qhBits3 = (pqh[l / 4 + 16] >> (2 * (l & 3))) & 3;
                    int q3 = qlNib3 | (qhBits3 << 4);
                    int i3 = nOff + l + 64;
                    if (i3 < blockEnd)
                        sum += input[b * QK_K + i3 - colBlockStart] * (d * psc[is_ + 4] * (q3 - 32));

                    int qlNib4 = (pql[l / 2 + 48] >> (4 * (l & 1))) & 0x0F;
                    int qhBits4 = (pqh[l / 4 + 24] >> (2 * (l & 3))) & 3;
                    int q4 = qlNib4 | (qhBits4 << 4);
                    int i4 = nOff + l + 96;
                    if (i4 < blockEnd)
                        sum += input[b * QK_K + i4 - colBlockStart] * (d * psc[is_ + 6] * (q4 - 32));
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
        float* vvBuf = stackalloc float[8];
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

            int i = curBlockStart;
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int qlByte = idx / 2;
                    int qlNib = (ql[qlByte] >> (4 * (idx & 1))) & 0x0F;
                    int qhByte = idx / 4;
                    int qhBits = (qh[qhByte] >> (2 * (idx & 3))) & 3;
                    int q = qlNib | (qhBits << 4);
                    vvBuf[sub] = d * scales[idx / 16] * (q - 32);
                }
                var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum256_Avx(Avx.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int qlByte = i / 2;
                int qlNib = (ql[qlByte] >> (4 * (i & 1))) & 0x0F;
                int qhByte = i / 4;
                int qhBits = (qh[qhByte] >> (2 * (i & 3))) & 3;
                int q = qlNib | (qhBits << 4);
                sum += pIn[i] * (d * scales[i / 16] * (q - 32));
            }
        }
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
        float* vvBuf = stackalloc float[8];
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

            int i = curBlockStart;
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int qlByte = idx / 2;
                    int qlNib = (ql[qlByte] >> (4 * (idx & 1))) & 0x0F;
                    int qhByte = idx / 4;
                    int qhBits = (qh[qhByte] >> (2 * (idx & 3))) & 3;
                    int q = qlNib | (qhBits << 4);
                    vvBuf[sub] = d * scales[idx / 16] * (q - 32);
                }
                var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum256_Avx(Fma.MultiplyAdd(vi, vv, Vector256<float>.Zero));
            }
            for (; i < blockEnd; i++)
            {
                int qlByte = i / 2;
                int qlNib = (ql[qlByte] >> (4 * (i & 1))) & 0x0F;
                int qhByte = i / 4;
                int qhBits = (qh[qhByte] >> (2 * (i & 3))) & 3;
                int q = qlNib | (qhBits << 4);
                sum += pIn[i] * (d * scales[i / 16] * (q - 32));
            }
        }
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
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ6K_Scalar(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
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
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ6K_AVX2(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
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
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ6K_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ6K_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ6K_FMA(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ6K_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

}
