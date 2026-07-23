using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{

    // VecDotQ5_0 � 5-bit block (QK=32)


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 22;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            uint qh = *(uint*)(block + 2);
            byte* qs = block + 6;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int h4 = ((int)(qh >> i) & 1) << 4;
                int half = QK / 2;
                int nib = (i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4);
                int q = nib | h4;
                sum += input[b * QK + i] * ((q - 16) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5_0_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 22;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        float* vvBuf = stackalloc float[8];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            uint qh = *(uint*)(block + 2);
            byte* qs = block + 6;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            int half = QK / 2;

            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int h4 = ((int)(qh >> idx) & 1) << 4;
                    int nib = (idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4);
                    vvBuf[sub] = ((nib | h4) - 16) * d;
                }
                var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, vw0));

                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + 8 + sub;
                    int h4 = ((int)(qh >> idx) & 1) << 4;
                    int nib = (idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4);
                    vvBuf[sub] = ((nib | h4) - 16) * d;
                }
                var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, vw1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int h4 = ((int)(qh >> idx) & 1) << 4;
                    int nib = (idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4);
                    vvBuf[sub] = ((nib | h4) - 16) * d;
                }
                var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, vw));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int h4 = ((int)(qh >> i) & 1) << 4;
                int nib = (i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4);
                int q = nib | h4;
                sum += pIn[i] * ((q - 16) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5_0_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 22;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        float* vvBuf = stackalloc float[8];
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            uint qh = *(uint*)(block + 2);
            byte* qs = block + 6;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            int half = QK / 2;

            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int h4 = ((int)(qh >> idx) & 1) << 4;
                    int nib = (idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4);
                    vvBuf[sub] = ((nib | h4) - 16) * d;
                }
                var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);

                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + 8 + sub;
                    int h4 = ((int)(qh >> idx) & 1) << 4;
                    int nib = (idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4);
                    vvBuf[sub] = ((nib | h4) - 16) * d;
                }
                var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int h4 = ((int)(qh >> idx) & 1) << 4;
                    int nib = (idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4);
                    vvBuf[sub] = ((nib | h4) - 16) * d;
                }
                var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Fma.MultiplyAdd(vi, vw, vacc0);
            }
            for (; i < blockEnd; i++)
            {
                int h4 = ((int)(qh >> i) & 1) << 4;
                int nib = (i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4);
                int q = nib | h4;
                sum += pIn[i] * ((q - 16) * d);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }


    // VecDotQ5_1 � 5-bit block with min (QK=32)


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5_1_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 24;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            float m = HalfToFloat_Scalar(*(ushort*)(block + 2));
            uint qh = *(uint*)(block + 4);
            byte* qs = block + 8;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int half = QK / 2;
                int q = ((i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)) | xh;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5_1_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 24;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        float* vvBuf = stackalloc float[8];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            float m = HalfToFloat_F16C(*(ushort*)(block + 2));
            uint qh = *(uint*)(block + 4);
            byte* qs = block + 8;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            var vm = Vector256.Create(m);
            int half = QK / 2;

            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int xh = (int)((qh >> idx) & 1) << 4;
                    int nib = ((idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4)) | xh;
                    vvBuf[sub] = nib * d + m;
                }
                var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, vw0));

                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + 8 + sub;
                    int xh = (int)((qh >> idx) & 1) << 4;
                    int nib = ((idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4)) | xh;
                    vvBuf[sub] = nib * d + m;
                }
                var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, vw1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int xh = (int)((qh >> idx) & 1) << 4;
                    int nib = ((idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4)) | xh;
                    vvBuf[sub] = nib * d + m;
                }
                var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, vw));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int nib = ((i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)) | xh;
                sum += pIn[i] * (nib * d + m);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5_1_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 24;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        float* vvBuf = stackalloc float[8];
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            float m = HalfToFloat_F16C(*(ushort*)(block + 2));
            uint qh = *(uint*)(block + 4);
            byte* qs = block + 8;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            int half = QK / 2;

            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int xh = (int)((qh >> idx) & 1) << 4;
                    int nib = ((idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4)) | xh;
                    vvBuf[sub] = nib * d + m;
                }
                var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);

                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + 8 + sub;
                    int xh = (int)((qh >> idx) & 1) << 4;
                    int nib = ((idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4)) | xh;
                    vvBuf[sub] = nib * d + m;
                }
                var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int xh = (int)((qh >> idx) & 1) << 4;
                    int nib = ((idx < half) ? (qs[idx] & 0x0F) : (qs[idx - half] >> 4)) | xh;
                    vvBuf[sub] = nib * d + m;
                }
                var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Fma.MultiplyAdd(vi, vw, vacc0);
            }
            for (; i < blockEnd; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int nib = ((i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)) | xh;
                sum += pIn[i] * (nib * d + m);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }


    // VecDotQ5K � K-quant 5-bit block (QK=256)


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 176;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            float min = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qh = block + 16;
            byte* qs = block + 48;
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
            {
                int sc = GetScaleMinK4_Scale_Scalar(i / 32, scales);
                int mn = GetScaleMinK4_Min_Scalar(i / 32, scales);
                int idx32 = i % 32;
                int group64 = i / 64;
                int half = (i % 64) / 32;
                int bitPos = group64 * 2 + half;
                int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                q5 |= hAdd;
                sum += input[b * QK_K + i - colBlockStart] * (sc * q5 * d - mn * min);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 176;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        float* vvBuf = stackalloc float[8];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            float min = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qh = block + 16;
            byte* qs = block + 48;
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            int i = curBlockStart;
            for (; i <= blockEnd - 16; i += 16)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int sc = GetScaleMinK4_Scale_Scalar(idx / 32, scales);
                    int mn = GetScaleMinK4_Min_Scalar(idx / 32, scales);
                    int idx32 = idx % 32;
                    int group64 = idx / 64;
                    int half = (idx % 64) / 32;
                    int bitPos = group64 * 2 + half;
                    int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                    int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                    q5 |= hAdd;
                    vvBuf[sub] = sc * q5 * d - mn * min;
                }
                var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, vw0));

                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + 8 + sub;
                    int sc = GetScaleMinK4_Scale_Scalar(idx / 32, scales);
                    int mn = GetScaleMinK4_Min_Scalar(idx / 32, scales);
                    int idx32 = idx % 32;
                    int group64 = idx / 64;
                    int half = (idx % 64) / 32;
                    int bitPos = group64 * 2 + half;
                    int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                    int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                    q5 |= hAdd;
                    vvBuf[sub] = sc * q5 * d - mn * min;
                }
                var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, vw1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int sc = GetScaleMinK4_Scale_Scalar(idx / 32, scales);
                    int mn = GetScaleMinK4_Min_Scalar(idx / 32, scales);
                    int idx32 = idx % 32;
                    int group64 = idx / 64;
                    int half = (idx % 64) / 32;
                    int bitPos = group64 * 2 + half;
                    int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                    int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                    q5 |= hAdd;
                    vvBuf[sub] = sc * q5 * d - mn * min;
                }
                var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, vw));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int sc = GetScaleMinK4_Scale_Scalar(i / 32, scales);
                int mn = GetScaleMinK4_Min_Scalar(i / 32, scales);
                int idx32 = i % 32;
                int group64 = i / 64;
                int half = (i % 64) / 32;
                int bitPos = group64 * 2 + half;
                int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                q5 |= hAdd;
                sum += pIn[i] * (sc * q5 * d - mn * min);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ5K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 176;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        float* vvBuf = stackalloc float[8];
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            float min = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qh = block + 16;
            byte* qs = block + 48;
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            int i = curBlockStart;
            for (; i <= blockEnd - 16; i += 16)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int sc = GetScaleMinK4_Scale_Scalar(idx / 32, scales);
                    int mn = GetScaleMinK4_Min_Scalar(idx / 32, scales);
                    int idx32 = idx % 32;
                    int group64 = idx / 64;
                    int half = (idx % 64) / 32;
                    int bitPos = group64 * 2 + half;
                    int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                    int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                    q5 |= hAdd;
                    vvBuf[sub] = sc * q5 * d - mn * min;
                }
                var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);

                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + 8 + sub;
                    int sc = GetScaleMinK4_Scale_Scalar(idx / 32, scales);
                    int mn = GetScaleMinK4_Min_Scalar(idx / 32, scales);
                    int idx32 = idx % 32;
                    int group64 = idx / 64;
                    int half = (idx % 64) / 32;
                    int bitPos = group64 * 2 + half;
                    int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                    int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                    q5 |= hAdd;
                    vvBuf[sub] = sc * q5 * d - mn * min;
                }
                var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int sc = GetScaleMinK4_Scale_Scalar(idx / 32, scales);
                    int mn = GetScaleMinK4_Min_Scalar(idx / 32, scales);
                    int idx32 = idx % 32;
                    int group64 = idx / 64;
                    int half = (idx % 64) / 32;
                    int bitPos = group64 * 2 + half;
                    int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                    int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                    q5 |= hAdd;
                    vvBuf[sub] = sc * q5 * d - mn * min;
                }
                var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, vw));
            }
            for (; i < blockEnd; i++)
            {
                int sc = GetScaleMinK4_Scale_Scalar(i / 32, scales);
                int mn = GetScaleMinK4_Min_Scalar(i / 32, scales);
                int idx32 = i % 32;
                int group64 = i / 64;
                int half = (i % 64) / 32;
                int bitPos = group64 * 2 + half;
                int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                q5 |= hAdd;
                sum += pIn[i] * (sc * q5 * d - mn * min);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }


    // ReadQ5 � dequantize from BinaryReader into Span<float>


    public static void ReadQ5_1_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 24;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            float m = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[2]));
            uint qh = Unsafe.ReadUnaligned<uint>(ref buf[4]);
            int valid = Math.Min(qk, n - blockStart);

            for (int i = 0; i < valid; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int half = qk / 2;
                int q = ((i < half) ? (buf[8 + i] & 0x0F) : (buf[8 + i - half] >> 4)) | xh;
                data[blockStart + i] = q * d + m;
            }
        }
    }

    public static void ReadQ5_0_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 22;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            uint qh = Unsafe.ReadUnaligned<uint>(ref buf[2]);
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int half = qk / 2;
                int nib = (j < half) ? (buf[6 + j] & 0x0F) : (buf[6 + j - half] >> 4);
                int h4 = ((int)(qh >> j) & 1) << 4;
                data[blockStart + j] = ((nib | h4) - 16) * d;
            }
        }
    }

    public static unsafe void ReadQ5_K_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 176;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);

            fixed (byte* pBuf = buf)
            {
                float d = HalfToFloat_Scalar(*(ushort*)pBuf);
                float min = HalfToFloat_Scalar(*(ushort*)(pBuf + 2));
                byte* scales = pBuf + 4;
                byte* qh = pBuf + 16;
                byte* qs = pBuf + 48;

                int valid = Math.Min(QK_K, n - blockStart);
                for (int i = 0; i < valid; i++)
                {
                    int sc = GetScaleMinK4_Scale_Scalar(i / 32, scales);
                    int mn = GetScaleMinK4_Min_Scalar(i / 32, scales);
                    int idx32 = i % 32;
                    int group64 = i / 64;
                    int half = (i % 64) / 32;
                    int bitPos = group64 * 2 + half;
                    int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                    int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                    q5 |= hAdd;
                    data[blockStart + i] = (sc * q5 * d) - (mn * min);
                }
            }
        }
    }


    // QuantizedMatMul � VecDot-based serial/parallel fallbacks


    public static unsafe void QuantizedMatMulQ5_0_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5_0_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5_0_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5_0_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ5_0_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5_0_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5_0_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5_0_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_0_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ5_0_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5_0_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5_0_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5_0_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_0_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ5_1_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5_1_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5_1_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5_1_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_1_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ5K_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5K_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5K_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5K_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5K_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ5_1_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5_1_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5_1_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5_1_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_1_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ5_1_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5_1_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5_1_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5_1_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_1_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ5K_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5K_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5K_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5K_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5K_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ5K_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5K_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ5K_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ5K_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5K_FMA(pInRow, rawWeights, col, K);
            });
        }
    }
}
