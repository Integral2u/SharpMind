using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    private static readonly float[] kvalues_iq4nl =
        { -127f, -104f, -83f, -65f, -49f, -35f, -22f, -10f, 1f, 13f, 25f, 38f, 53f, 69f, 89f, 113f };

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_0_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                var w0 = Avx.Multiply(Vector256.Create(
                    (float)(((qs[i / 2] >> (4 * (i % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 4) / 2] >> (4 * ((i + 4) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 5) / 2] >> (4 * ((i + 5) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 6) / 2] >> (4 * ((i + 6) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 7) / 2] >> (4 * ((i + 7) % 2))) & 0x0F) - 8)
                ), vd);
                var w1 = Avx.Multiply(Vector256.Create(
                    (float)(((qs[(i + 8) / 2] >> (4 * ((i + 8) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 9) / 2] >> (4 * ((i + 9) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 10) / 2] >> (4 * ((i + 10) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 11) / 2] >> (4 * ((i + 11) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 12) / 2] >> (4 * ((i + 12) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 13) / 2] >> (4 * ((i + 13) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 14) / 2] >> (4 * ((i + 14) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 15) / 2] >> (4 * ((i + 15) % 2))) & 0x0F) - 8)
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w0));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 8]), w1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var w = Avx.Multiply(Vector256.Create(
                    (float)(((qs[i / 2] >> (4 * (i % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 4) / 2] >> (4 * ((i + 4) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 5) / 2] >> (4 * ((i + 5) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 6) / 2] >> (4 * ((i + 6) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 7) / 2] >> (4 * ((i + 7) % 2))) & 0x0F) - 8)
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_0_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            int i = 0;
            for (; i <= blockEnd - 4; i += 4)
            {
                float v0 = ((qs[i / 2] >> (4 * (i % 2))) & 0x0F) - 8;
                float v1 = ((qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F) - 8;
                float v2 = ((qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F) - 8;
                float v3 = ((qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F) - 8;
                var vv = Vector128.Create(v0, v1, v2, v3) * Vector128.Create(d);
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum128_Sse(Sse.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_1_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            float m = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* qs = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_1_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            float m = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* qs = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            var vm = Vector256.Create(m);

            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                float v0 = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                float v1 = (qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F;
                float v2 = (qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F;
                float v3 = (qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F;
                float v4 = (qs[(i + 4) / 2] >> (4 * ((i + 4) % 2))) & 0x0F;
                float v5 = (qs[(i + 5) / 2] >> (4 * ((i + 5) % 2))) & 0x0F;
                float v6 = (qs[(i + 6) / 2] >> (4 * ((i + 6) % 2))) & 0x0F;
                float v7 = (qs[(i + 7) / 2] >> (4 * ((i + 7) % 2))) & 0x0F;
                var vv = Avx.Multiply(Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7), vd);
                vv = Avx.Add(vv, vm);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum256_Avx(Avx.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_1_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            float m = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* qs = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector128.Create(d);
            var vm = Vector128.Create(m);

            int i = 0;
            for (; i <= blockEnd - 4; i += 4)
            {
                float v0 = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                float v1 = (qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F;
                float v2 = (qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F;
                float v3 = (qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F;
                var vv = Sse.Add(Sse.Multiply(Vector128.Create(v0, v1, v2, v3), vd), vm);
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum128_Sse(Sse.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_NL_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int nib = (qs[i >> 1] >> (4 * (i & 1))) & 0x0F;
                sum += input[b * QK + i] * (d * kvalues_iq4nl[nib]);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_NL_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                var w0 = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(qs[i>>1] >> (4*(i&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+1)>>1] >> (4*((i+1)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+2)>>1] >> (4*((i+2)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+3)>>1] >> (4*((i+3)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+4)>>1] >> (4*((i+4)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+5)>>1] >> (4*((i+5)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+6)>>1] >> (4*((i+6)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+7)>>1] >> (4*((i+7)&1))) & 0x0F]
                ), vd);
                var w1 = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(qs[(i+8)>>1] >> (4*((i+8)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+9)>>1] >> (4*((i+9)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+10)>>1] >> (4*((i+10)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+11)>>1] >> (4*((i+11)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+12)>>1] >> (4*((i+12)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+13)>>1] >> (4*((i+13)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+14)>>1] >> (4*((i+14)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+15)>>1] >> (4*((i+15)&1))) & 0x0F]
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w0));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 8]), w1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var w = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(qs[i>>1] >> (4*(i&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+1)>>1] >> (4*((i+1)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+2)>>1] >> (4*((i+2)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+3)>>1] >> (4*((i+3)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+4)>>1] >> (4*((i+4)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+5)>>1] >> (4*((i+5)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+6)>>1] >> (4*((i+6)&1))) & 0x0F],
                    kvalues_iq4nl[(qs[(i+7)>>1] >> (4*((i+7)&1))) & 0x0F]
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int nib = (qs[i >> 1] >> (4 * (i & 1))) & 0x0F;
                sum += pIn[i] * (d * kvalues_iq4nl[nib]);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 12;
            float dSuper = HalfToFloat_Scalar(*(ushort*)(block + 140));
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 142));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s0 = GetScaleMinK4_Scale_Scalar(isc * 2, scales);
                    float m0 = GetScaleMinK4_Min_Scalar(isc * 2, scales);
                    for (int l = 0; l < 16 && basePos + l < blockEnd; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = idx / 2;
                        int qsShift = (idx & 1) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += input[b * QK_K + idx - colBlockStart] * (s0 * v * dSuper - m0 * minSuper);
                    }

                    float s1 = GetScaleMinK4_Scale_Scalar(isc * 2 + 1, scales);
                    float m1 = GetScaleMinK4_Min_Scalar(isc * 2 + 1, scales);
                    for (int l = 0; l < 16 && basePos + 16 + l < blockEnd; l++)
                    {
                        int idx = basePos + 16 + l;
                        int qsByte = idx / 2;
                        int qsShift = (idx & 1) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += input[b * QK_K + idx - colBlockStart] * (s1 * v * dSuper - m1 * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        float* vvBuf = stackalloc float[8];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 12;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 140));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 142));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s0 = GetScaleMinK4_Scale_Scalar(isc * 2, scales);
                    float m0 = GetScaleMinK4_Min_Scalar(isc * 2, scales);
                    var vs0 = Vector256.Create(s0 * dSuper);
                    var vm0 = Vector256.Create(m0 * minSuper);

                    int subRem = Math.Min(16, blockEnd - basePos);
                    int l = 0;
                    for (; l <= subRem - 8; l += 8)
                    {
                        for (int sub = 0; sub < 8; sub++)
                        {
                            int idx = basePos + l + sub;
                            int qsByte = idx / 2;
                            int qsShift = (idx & 1) * 4;
                            vvBuf[sub] = (qs[qsByte] >> qsShift) & 0x0F;
                        }
                        var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                        var vi = Vector256.LoadUnsafe(ref pIn[basePos + l]);
                        var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0), vm0));
                        sum += MathHelpers.HSum256_Avx(res);
                    }
                    for (; l < subRem; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = idx / 2;
                        int qsShift = (idx & 1) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += pIn[idx] * (s0 * v * dSuper - m0 * minSuper);
                    }

                    float s1 = GetScaleMinK4_Scale_Scalar(isc * 2 + 1, scales);
                    float m1 = GetScaleMinK4_Min_Scalar(isc * 2 + 1, scales);
                    var vs1 = Vector256.Create(s1 * dSuper);
                    var vm1 = Vector256.Create(m1 * minSuper);

                    int bPos1 = basePos + 16;
                    int subRem2 = Math.Min(16, blockEnd - bPos1);
                    int l2 = 0;
                    for (; l2 <= subRem2 - 8; l2 += 8)
                    {
                        for (int sub2 = 0; sub2 < 8; sub2++)
                        {
                            int idx = bPos1 + l2 + sub2;
                            int qsByte = idx / 2;
                            int qsShift = (idx & 1) * 4;
                            vvBuf[sub2] = (qs[qsByte] >> qsShift) & 0x0F;
                        }
                        var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                        var vi = Vector256.LoadUnsafe(ref pIn[bPos1 + l2]);
                        var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1), vm1));
                        sum += MathHelpers.HSum256_Avx(res);
                    }
                    for (; l2 < subRem2; l2++)
                    {
                        int idx = bPos1 + l2;
                        int qsByte = idx / 2;
                        int qsShift = (idx & 1) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += pIn[idx] * (s1 * v * dSuper - m1 * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        float* vvBuf = stackalloc float[8];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 12;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 140));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 142));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s0 = GetScaleMinK4_Scale_Scalar(isc * 2, scales);
                    float m0 = GetScaleMinK4_Min_Scalar(isc * 2, scales);
                    var vs0 = Vector256.Create(s0 * dSuper);
                    var vm0 = Vector256.Create(m0 * minSuper);

                    int subRem = Math.Min(16, blockEnd - basePos);
                    int l = 0;
                    for (; l <= subRem - 8; l += 8)
                    {
                        for (int sub = 0; sub < 8; sub++)
                        {
                            int idx = basePos + l + sub;
                            int qsByte = idx / 2;
                            int qsShift = (idx & 1) * 4;
                            vvBuf[sub] = (qs[qsByte] >> qsShift) & 0x0F;
                        }
                        var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                        var vi = Vector256.LoadUnsafe(ref pIn[basePos + l]);
                        var vw = Avx.Subtract(Avx.Multiply(vv, vs0), vm0);
                        var res = Avx.Multiply(vi, vw);
                        sum += MathHelpers.HSum256_Avx(res);
                    }
                    for (; l < subRem; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = idx / 2;
                        int qsShift = (idx & 1) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += pIn[idx] * (s0 * v * dSuper - m0 * minSuper);
                    }

                    float s1 = GetScaleMinK4_Scale_Scalar(isc * 2 + 1, scales);
                    float m1 = GetScaleMinK4_Min_Scalar(isc * 2 + 1, scales);
                    var vs1 = Vector256.Create(s1 * dSuper);
                    var vm1 = Vector256.Create(m1 * minSuper);

                    int bPos1 = basePos + 16;
                    int subRem2 = Math.Min(16, blockEnd - bPos1);
                    int l2 = 0;
                    for (; l2 <= subRem2 - 8; l2 += 8)
                    {
                        for (int sub2 = 0; sub2 < 8; sub2++)
                        {
                            int idx = bPos1 + l2 + sub2;
                            int qsByte = idx / 2;
                            int qsShift = (idx & 1) * 4;
                            vvBuf[sub2] = (qs[qsByte] >> qsShift) & 0x0F;
                        }
                        var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                        var vi = Vector256.LoadUnsafe(ref pIn[bPos1 + l2]);
                        var vw = Avx.Subtract(Avx.Multiply(vv, vs1), vm1);
                        var res = Avx.Multiply(vi, vw);
                        sum += MathHelpers.HSum256_Avx(res);
                    }
                    for (; l2 < subRem2; l2++)
                    {
                        int idx = bPos1 + l2;
                        int qsByte = idx / 2;
                        int qsShift = (idx & 1) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += pIn[idx] * (s1 * v * dSuper - m1 * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulQ4_0_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_0_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_0_Scalar(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_1_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_1_Scalar(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_1_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_NL_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_NL_Scalar(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_NL_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4K_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4K_Scalar(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4K_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void ReadQ4_0_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 18;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int nib = (buf[2 + j / 2] >> ((j & 1) * 4)) & 0x0F;
                data[blockStart + j] = (nib - 8) * d;
            }
        }
    }

    public static unsafe void ReadQ4_1_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 20;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            float m = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[2]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int q = (buf[4 + j / 2] >> ((j & 1) * 4)) & 0x0F;
                data[blockStart + j] = q * d + m;
            }
        }
    }

    public static unsafe void ReadQ4_NL_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 18;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int nib = (buf[2 + (j >> 1)] >> (4 * (j & 1))) & 0x0F;
                data[blockStart + j] = d * kvalues_iq4nl[nib];
            }
        }
    }

    public static unsafe void ReadQ4K_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 144;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);
            float dSuper = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[140]));
            float minSuper = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[142]));

            fixed (byte* pBuf = buf)
            {
                byte* scales = pBuf;
                byte* qs = pBuf + 12;
                for (int i = 0; i < QK_K && blockStart + i < n; i++)
                {
                    int sub = i / 32;
                    float s = GetScaleMinK4_Scale_Scalar(sub, scales);
                    float m = GetScaleMinK4_Min_Scalar(sub, scales);
                    int qsByte = i / 2;
                    int qsShift = (i & 1) * 4;
                    int v = (qs[qsByte] >> qsShift) & 0x0F;
                    data[blockStart + i] = s * v * dSuper - m * minSuper;
                }
            }
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_0_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_0_AVX2(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_0_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Serial_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_0_SSE(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Parallel_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_0_SSE(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_0_SSE(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_1_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_1_AVX2(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_1_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Serial_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_1_SSE(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Parallel_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_1_SSE(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_1_SSE(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_NL_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_NL_AVX2(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_NL_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4K_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4K_AVX2(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4K_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4K_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4K_FMA(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4K_FMA(pInRow, rawWeights, col, K);
            });
        }
    }
}
