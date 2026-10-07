using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    // Eight consecutive Q3_K codes for block-relative positions idx..idx+7,
    // already folded into the -4..3 signed range: the two-bit nibble (packed the
    // same way as Q2_K, so one widened load per group) plus the high-mask bit
    // that re-adds 4 when set. Replaces the per-element divide/modulo that used
    // to fill a stack buffer before every vector multiply.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> Q3KCodes8(byte* qs, byte* hmask, int idx)
    {
        var w = Avx2.ConvertToVector256Int32(qs + ((idx >> 7) << 5) + (idx & 31));
        w = Avx2.And(Avx2.ShiftRightLogical(w, (byte)((((idx >> 5) & 3) << 1))), Vector256.Create(3));
        var h = Avx2.And(Avx2.ShiftRightLogical(Avx2.ConvertToVector256Int32(hmask + (idx & 31)), (byte)(idx >> 5)), Vector256.Create(1));
        return Avx.Add(Avx.ConvertToVector256Single(w), Avx.Add(Avx.Multiply(Avx.ConvertToVector256Single(h), Vector256.Create(4f)), Vector256.Create(-4f)));
    }

    // Same eight codes with the byte addresses and both plane shifts already
    // resolved by the caller, which holds them constant across a whole 32-value
    // group so four chains can run without re-deriving them per iteration.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> Q3KCodesAt(byte* q, byte qsh, byte* hm, byte hsh)
    {
        var w = Avx2.And(Avx2.ShiftRightLogical(Avx2.ConvertToVector256Int32(q), qsh), Vector256.Create(3));
        var h = Avx2.And(Avx2.ShiftRightLogical(Avx2.ConvertToVector256Int32(hm), hsh), Vector256.Create(1));
        return Avx.Add(Avx.ConvertToVector256Single(w), Avx.Add(Avx.Multiply(Avx.ConvertToVector256Single(h), Vector256.Create(4f)), Vector256.Create(-4f)));
    }

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
            for (; i <= blockEnd - 8 && (i & 7) != 0; i++)
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
            for (; i <= blockEnd - 8; i += 8)
            {
                float k = dAll * (sc8[i >> 4] - 32);
                var vv = Avx.Multiply(Vector256.Create(k), Q3KCodes8(qs, hmask, i));
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

        var vacc = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        var vacc2 = Vector256<float>.Zero;
        var vacc3 = Vector256<float>.Zero;
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
            for (; i <= blockEnd - 8 && (i & 7) != 0; i++)
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
            // A whole 32-value group shares one qs shift, one hmask shift and two
            // scales, so it decodes as four independent chains. The packed-byte ->
            // shift -> mask -> convert -> scale -> code-path chain is ~30 cycles
            // and a single accumulator ran all of them back to back.
            for (; (i & 31) == 0 && i + 32 <= blockEnd; i += 32)
            {
                float k0 = dAll * (sc8[i >> 4] - 32);
                float k1 = dAll * (sc8[(i + 16) >> 4] - 32);
                var ks0 = Vector256.Create(k0);
                var ks1 = Vector256.Create(k1);
                byte* q = qs + ((i >> 7) << 5) + (i & 31);
                byte qsh = (byte)(((i >> 5) & 3) << 1);
                byte hsh = (byte)(i >> 5);
                var u0 = Avx.Multiply(ks0, Q3KCodesAt(q, qsh, hmask, hsh));
                var u1 = Avx.Multiply(ks0, Q3KCodesAt(q + 8, qsh, hmask + 8, hsh));
                var u2 = Avx.Multiply(ks1, Q3KCodesAt(q + 16, qsh, hmask + 16, hsh));
                var u3 = Avx.Multiply(ks1, Q3KCodesAt(q + 24, qsh, hmask + 24, hsh));
                vacc = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), u0, vacc);
                vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i + 8]), u1, vacc1);
                vacc2 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i + 16]), u2, vacc2);
                vacc3 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i + 24]), u3, vacc3);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                float k = dAll * (sc8[i >> 4] - 32);
                var vv = Avx.Multiply(Vector256.Create(k), Q3KCodes8(qs, hmask, i));
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
        sum += MathHelpers.HSum256_Avx(Avx.Add(Avx.Add(vacc, vacc1), Avx.Add(vacc2, vacc3)));
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
