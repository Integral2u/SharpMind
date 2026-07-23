using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ3K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 110;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        byte* scaleBuf = stackalloc byte[16];

        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* hmask = block;
            byte* qs = block + 32;
            float dAll = HalfToFloat_Scalar(*(ushort*)(block + 108));

            uint* aux = (uint*)scaleBuf;
            aux[0] = *(uint*)(block + 96);
            aux[1] = *(uint*)(block + 100);
            aux[2] = *(uint*)(block + 104);
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            sbyte* sc8 = (sbyte*)scaleBuf;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
            {
                int qsByte = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int s2 = (qs[qsByte] >> qsShift) & 3;
                int hBit = (hmask[i % 32] >> (i / 32)) & 1;
                int actual = s2 - (hBit == 0 ? 4 : 0);
                float val = dAll * (sc8[i / 16] - 32) * actual;
                sum += input[b * QK_K + i - colBlockStart] * val;
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ3K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 110;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        byte* scaleBuf = stackalloc byte[16];

        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        float* vvBuf = stackalloc float[8];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* hmask = block;
            byte* qs = block + 32;
            float dAll = HalfToFloat_F16C(*(ushort*)(block + 108));

            uint* aux = (uint*)scaleBuf;
            aux[0] = *(uint*)(block + 96);
            aux[1] = *(uint*)(block + 100);
            aux[2] = *(uint*)(block + 104);
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            sbyte* sc8 = (sbyte*)scaleBuf;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            int i = curBlockStart;
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int qsByte = (idx / 128) * 32 + (idx % 32);
                    int qsShift = ((idx % 128) / 32) * 2;
                    int s2 = (qs[qsByte] >> qsShift) & 3;
                    int hBit = (hmask[idx % 32] >> (idx / 32)) & 1;
                    int actual = s2 - (hBit == 0 ? 4 : 0);
                    vvBuf[sub] = dAll * (sc8[idx / 16] - 32) * actual;
                }
                var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum256_Avx(Avx.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int idx = i;
                int qsByte = (idx / 128) * 32 + (idx % 32);
                int qsShift = ((idx % 128) / 32) * 2;
                int s2 = (qs[qsByte] >> qsShift) & 3;
                int hBit = (hmask[idx % 32] >> (idx / 32)) & 1;
                int actual = s2 - (hBit == 0 ? 4 : 0);
                float val = dAll * (sc8[idx / 16] - 32) * actual;
                sum += pIn[i] * val;
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ3K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 110;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        byte* scaleBuf = stackalloc byte[16];

        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        float* vvBuf = stackalloc float[8];
        var vacc = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* hmask = block;
            byte* qs = block + 32;
            float dAll = HalfToFloat_F16C(*(ushort*)(block + 108));

            uint* aux = (uint*)scaleBuf;
            aux[0] = *(uint*)(block + 96);
            aux[1] = *(uint*)(block + 100);
            aux[2] = *(uint*)(block + 104);
            uint tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            sbyte* sc8 = (sbyte*)scaleBuf;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            int i = curBlockStart;
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int qsByte = (idx / 128) * 32 + (idx % 32);
                    int qsShift = ((idx % 128) / 32) * 2;
                    int s2 = (qs[qsByte] >> qsShift) & 3;
                    int hBit = (hmask[idx % 32] >> (idx / 32)) & 1;
                    int actual = s2 - (hBit == 0 ? 4 : 0);
                    vvBuf[sub] = dAll * (sc8[idx / 16] - 32) * actual;
                }
                var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                vacc = Fma.MultiplyAdd(vi, vv, vacc);
            }
            for (; i < blockEnd; i++)
            {
                int idx = i;
                int qsByte = (idx / 128) * 32 + (idx % 32);
                int qsShift = ((idx % 128) / 32) * 2;
                int s2 = (qs[qsByte] >> qsShift) & 3;
                int hBit = (hmask[idx % 32] >> (idx / 32)) & 1;
                int actual = s2 - (hBit == 0 ? 4 : 0);
                float val = dAll * (sc8[idx / 16] - 32) * actual;
                sum += pIn[i] * val;
            }
        }
        sum += MathHelpers.HSum256_Avx(vacc);
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulQ3K_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ3K_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ3K_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ3K_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ3K_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void ReadQ3_K_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 110;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];
        uint* pAux = stackalloc uint[4];

        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);

            float dAll = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[108]));

            pAux[0] = Unsafe.ReadUnaligned<uint>(ref buf[96]);
            pAux[1] = Unsafe.ReadUnaligned<uint>(ref buf[100]);
            pAux[2] = Unsafe.ReadUnaligned<uint>(ref buf[104]);
            uint tmp2 = pAux[2];
            pAux[2] = ((pAux[0] >> 4) & kmask2) | (((tmp2 >> 4) & kmask1) << 4);
            pAux[3] = ((pAux[1] >> 4) & kmask2) | (((tmp2 >> 6) & kmask1) << 4);
            pAux[0] = (pAux[0] & kmask2) | (((tmp2 >> 0) & kmask1) << 4);
            pAux[1] = (pAux[1] & kmask2) | (((tmp2 >> 2) & kmask1) << 4);

            sbyte* scales = (sbyte*)pAux;

            int valid = Math.Min(QK_K, n - blockStart);
            int idx = 0;
            int qOff = 32;

            for (int half = 0; half < 2; half++)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    float s1 = scales[idx] - 32;
                    float s2 = scales[idx + 1] - 32;

                    int gIdx1 = idx;
                    int gIdx2 = idx + 1;

                    int lim1 = Math.Min(16, valid - gIdx1 * 16);
                    for (int l = 0; l < lim1; l++)
                    {
                        int relPos = gIdx1 * 16 + l;
                        int hmBit = (buf[relPos % 32] >> (relPos / 32)) & 1;
                        int q2 = (buf[qOff + l] >> shift) & 3;
                        data[blockStart + gIdx1 * 16 + l] = (s1 * (q2 - (hmBit != 0 ? 0 : 4))) * dAll;
                    }

                    int lim2 = Math.Min(16, valid - gIdx2 * 16);
                    for (int l = 0; l < lim2; l++)
                    {
                        int relPos = gIdx2 * 16 + l;
                        int hmBit = (buf[relPos % 32] >> (relPos / 32)) & 1;
                        int q2 = (buf[qOff + 16 + l] >> shift) & 3;
                        data[blockStart + gIdx2 * 16 + l] = (s2 * (q2 - (hmBit != 0 ? 0 : 4))) * dAll;
                    }

                    idx += 2;
                    shift += 2;
                }
                qOff += 32;
            }
        }
    }

    public static unsafe void QuantizedMatMulQ3K_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ3K_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ3K_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ3K_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ3K_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ3K_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ3K_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ3K_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ3K_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ3K_FMA(pInRow, rawWeights, col, K);
            });
        }
    }
}
