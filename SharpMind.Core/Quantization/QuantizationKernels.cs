using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    internal const int QK_K = 256;
    internal const int QK   = 32;

    // ═══════════════════════════════════════════════════════════════════════
    // HalfToFloat — FP16 → FP32
    // ═══════════════════════════════════════════════════════════════════════

    public static unsafe float HalfToFloat_F16C(ushort half)
    {
        int exp5 = (half >> 10) & 0x1F;

        if (exp5 == 0)
        {
            uint mant10 = (uint)(half & 0x3FF);
            if (mant10 == 0)
                return (half & 0x8000) == 0 ? 0f : -0f;

            int lz = BitOperations.LeadingZeroCount(mant10);
            int k = 31 - lz;
            uint e = (uint)(k + 103);
            uint m = (mant10 - (1u << k)) << (23 - k);
            uint bitsSub = ((uint)(half & 0x8000) << 16) | (e << 23) | m;
            return *(float*)&bitsSub;
        }

        if (exp5 == 31)
        {
            if ((half & 0x3FF) == 0)
                return (half & 0x8000) == 0 ? float.PositiveInfinity : float.NegativeInfinity;
            return float.NaN;
        }

        uint eBits = (uint)(exp5 + 112);
        uint mMant = (uint)(half & 0x3FF) << 13;
        uint bitsNrm = ((uint)(half & 0x8000) << 16) | (eBits << 23) | mMant;
        return *(float*)&bitsNrm;
    }

    internal static float HalfToFloat_Scalar(ushort half) => HalfToFloat_F16C(half);

    // ═══════════════════════════════════════════════════════════════════════
    // GetScaleMinK4 — 6-bit K-quant scale/min unpacking
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe byte GetScaleMinK4_Scale_Scalar(int j, byte* scales)
    {
        if (j < 4)
            return (byte)(scales[j] & 0x3F);
        return (byte)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
    }

    internal static unsafe byte GetScaleMinK4_Min_Scalar(int j, byte* scales)
    {
        if (j < 4)
            return (byte)(scales[j + 4] & 0x3F);
        return (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ3K — 3-bit K-quant
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ3K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 110;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        byte* scaleBuf = stackalloc byte[16];

        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* hmask  = block;
            byte* qs     = block + 32;
            float dAll   = HalfToFloat_Scalar(*(ushort*)(block + 108));

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

            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            for (int i = 0; i < blockEnd; i++)
            {
                int qsByte  = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int s2      = (qs[qsByte] >> qsShift) & 3;
                int hBit    = (hmask[i % 32] >> (i / 32)) & 1;
                int actual  = s2 - (hBit == 0 ? 4 : 0);
                float val   = dAll * (sc8[i / 16] - 32) * actual;
                sum += input[b * QK_K + i] * val;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ3K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 110;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        byte* scaleBuf = stackalloc byte[16];

        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        float* vvBuf = stackalloc float[8];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* hmask  = block;
            byte* qs     = block + 32;
            float dAll   = HalfToFloat_Scalar(*(ushort*)(block + 108));

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

            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int qsByte  = (idx / 128) * 32 + (idx % 32);
                    int qsShift = ((idx % 128) / 32) * 2;
                    int s2      = (qs[qsByte] >> qsShift) & 3;
                    int hBit    = (hmask[idx % 32] >> (idx / 32)) & 1;
                    int actual  = s2 - (hBit == 0 ? 4 : 0);
                    vvBuf[sub]  = dAll * (sc8[idx / 16] - 32) * actual;
                }
                var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum256_Avx(Avx.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int idx = i;
                int qsByte  = (idx / 128) * 32 + (idx % 32);
                int qsShift = ((idx % 128) / 32) * 2;
                int s2      = (qs[qsByte] >> qsShift) & 3;
                int hBit    = (hmask[idx % 32] >> (idx / 32)) & 1;
                int actual  = s2 - (hBit == 0 ? 4 : 0);
                float val   = dAll * (sc8[idx / 16] - 32) * actual;
                sum += pIn[i] * val;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ3K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 110;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        byte* scaleBuf = stackalloc byte[16];

        const uint kmask1 = 0x03030303u;
        const uint kmask2 = 0x0f0f0f0fu;

        float* vvBuf = stackalloc float[8];
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* hmask  = block;
            byte* qs     = block + 32;
            float dAll   = HalfToFloat_Scalar(*(ushort*)(block + 108));

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

            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;

            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int qsByte  = (idx / 128) * 32 + (idx % 32);
                    int qsShift = ((idx % 128) / 32) * 2;
                    int s2      = (qs[qsByte] >> qsShift) & 3;
                    int hBit    = (hmask[idx % 32] >> (idx / 32)) & 1;
                    int actual  = s2 - (hBit == 0 ? 4 : 0);
                    vvBuf[sub]  = dAll * (sc8[idx / 16] - 32) * actual;
                }
                var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum256_Avx(Fma.MultiplyAdd(vi, vv, Vector256<float>.Zero));
            }
            for (; i < blockEnd; i++)
            {
                int idx = i;
                int qsByte  = (idx / 128) * 32 + (idx % 32);
                int qsShift = ((idx % 128) / 32) * 2;
                int s2      = (qs[qsByte] >> qsShift) & 3;
                int hBit    = (hmask[idx % 32] >> (idx / 32)) & 1;
                int actual  = s2 - (hBit == 0 ? 4 : 0);
                float val   = dAll * (sc8[idx / 16] - 32) * actual;
                sum += pIn[i] * val;
            }
        }
        return (float)sum;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ4K — 4-bit K-quant
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ4K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block    = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float dSuper   = HalfToFloat_Scalar(*(ushort*)block);
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales   = block + 4;
            byte* qs       = block + 16;

            int idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0 = GetScaleMinK4_Scale_Scalar(idx + 0, scales);
                byte m0  = GetScaleMinK4_Min_Scalar(idx + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(idx + 1, scales);
                byte m1  = GetScaleMinK4_Min_Scalar(idx + 1, scales);
                float d1 = dSuper * sc0;
                float m1v = minSuper * m0;
                float d2 = dSuper * sc1;
                float m2v = minSuper * m1;

                int qIdx = (j / 64) * 32;
                int remaining = Math.Min(64, inFeatures - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);
                for (int l = 0; l < halfRem; l++)
                {
                    int pos = b * QK_K + j + l;
                    sum += input[pos] * (d1 * (qs[qIdx + l] & 0x0F) - m1v);
                }
                for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                {
                    int pos = b * QK_K + j + 32 + l;
                    sum += input[pos] * (d2 * (qs[qIdx + l] >> 4) - m2v);
                }
                idx += 2;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ4K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block    = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float dSuper   = HalfToFloat_Scalar(*(ushort*)block);
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales   = block + 4;
            byte* qs       = block + 16;

            float* pIn = input + b * QK_K;
            int idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0 = GetScaleMinK4_Scale_Scalar(idx + 0, scales);
                byte m0  = GetScaleMinK4_Min_Scalar(idx + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(idx + 1, scales);
                byte m1  = GetScaleMinK4_Min_Scalar(idx + 1, scales);
                var vs0 = Vector256.Create((float)sc0);
                var vm0 = Vector256.Create((float)m0);
                var vs1 = Vector256.Create((float)sc1);
                var vm1 = Vector256.Create((float)m1);
                var vdSuper = Vector256.Create(dSuper);
                var vminSuper = Vector256.Create(minSuper);
                
                var vs0_d = Avx.Multiply(vs0, vdSuper);
                var vm0_min = Avx.Multiply(vm0, vminSuper);
                var vs1_d = Avx.Multiply(vs1, vdSuper);
                var vm1_min = Avx.Multiply(vm1, vminSuper);
                
                int qIdx = (j / 64) * 32;
                int remaining = Math.Min(64, inFeatures - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);

                if (halfRem >= 8)
                {
                    var vacc1 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= halfRem - 8; l += 8)
                    {
                        float v0 = qs[qIdx + l] & 0x0F;
                        float v1 = qs[qIdx + l + 1] & 0x0F;
                        float v2 = qs[qIdx + l + 2] & 0x0F;
                        float v3 = qs[qIdx + l + 3] & 0x0F;
                        float v4 = qs[qIdx + l + 4] & 0x0F;
                        float v5 = qs[qIdx + l + 5] & 0x0F;
                        float v6 = qs[qIdx + l + 6] & 0x0F;
                        float v7 = qs[qIdx + l + 7] & 0x0F;
                                var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                                var vi = Vector256.LoadUnsafe(ref pIn[j + l]);
                                var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0_d), vm0_min));
                                vacc1 = Avx.Add(vacc1, res);
                            }
                            sum += MathHelpers.HSum256_Avx(vacc1);
                            for (; l < halfRem; l++)
                            {
                                int pos = b * QK_K + j + l;
                                sum += input[pos] * ((sc0 * (qs[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper));
                            }
                        }
                        else
                        {
                            for (int l = 0; l < halfRem; l++)
                            {
                                int pos = b * QK_K + j + l;
                                sum += input[pos] * ((sc0 * (qs[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper));
                            }
                        }
                        
                        int half2End = Math.Min(32, remaining - 32);
                        if (half2End > 0 && half2End >= 8)
                        {
                            var vacc2 = Vector256<float>.Zero;
                            int l = 0;
                            for (; l <= half2End - 8; l += 8)
                            {
                                float v0 = qs[qIdx + l] >> 4;
                                float v1 = qs[qIdx + l + 1] >> 4;
                                float v2 = qs[qIdx + l + 2] >> 4;
                                float v3 = qs[qIdx + l + 3] >> 4;
                                float v4 = qs[qIdx + l + 4] >> 4;
                                float v5 = qs[qIdx + l + 5] >> 4;
                                float v6 = qs[qIdx + l + 6] >> 4;
                                float v7 = qs[qIdx + l + 7] >> 4;
                                var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                                var vi = Vector256.LoadUnsafe(ref pIn[j + 32 + l]);
                                var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1_d), vm1_min));
                                vacc2 = Avx.Add(vacc2, res);
                            }
                            sum += MathHelpers.HSum256_Avx(vacc2);
                            for (; l < half2End; l++)
                            {
                                int pos = b * QK_K + j + 32 + l;
                                sum += input[pos] * ((sc1 * (qs[qIdx + l] >> 4) * dSuper) - (m1 * minSuper));
                            }
                        }
                        else
                        {
                            for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                            {
                                int pos = b * QK_K + j + 32 + l;
                                sum += input[pos] * ((sc1 * (qs[qIdx + l] >> 4) * dSuper) - (m1 * minSuper));
                            }
                        }
                idx += 2;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ4K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block    = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float dSuper   = HalfToFloat_Scalar(*(ushort*)block);
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales   = block + 4;
            byte* qs       = block + 16;

            float* pIn = input + b * QK_K;
            int idx = 0;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0 = GetScaleMinK4_Scale_Scalar(idx + 0, scales);
                byte m0  = GetScaleMinK4_Min_Scalar(idx + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(idx + 1, scales);
                byte m1  = GetScaleMinK4_Min_Scalar(idx + 1, scales);
                var vs0 = Vector256.Create((float)sc0);
                var vm0 = Vector256.Create((float)m0);
                var vs1 = Vector256.Create((float)sc1);
                var vm1 = Vector256.Create((float)m1);
                var vdSuper = Vector256.Create(dSuper);
                var vminSuper = Vector256.Create(minSuper);
                
                var vs0_d = Avx.Multiply(vs0, vdSuper);
                var vm0_min = Avx.Multiply(vm0, vminSuper);
                var vs1_d = Avx.Multiply(vs1, vdSuper);
                var vm1_min = Avx.Multiply(vm1, vminSuper);
                
                int qIdx = (j / 64) * 32;
                int remaining = Math.Min(64, inFeatures - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);

                if (halfRem >= 8)
                {
                    var vacc1 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= halfRem - 8; l += 8)
                    {
                        float v0 = qs[qIdx + l] & 0x0F;
                        float v1 = qs[qIdx + l + 1] & 0x0F;
                        float v2 = qs[qIdx + l + 2] & 0x0F;
                        float v3 = qs[qIdx + l + 3] & 0x0F;
                        float v4 = qs[qIdx + l + 4] & 0x0F;
                        float v5 = qs[qIdx + l + 5] & 0x0F;
                        float v6 = qs[qIdx + l + 6] & 0x0F;
                        float v7 = qs[qIdx + l + 7] & 0x0F;
                                var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                                var vi = Vector256.LoadUnsafe(ref pIn[j + l]);
                                var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0_d), vm0_min));
                                vacc1 = Avx.Add(vacc1, res);
                            }
                            sum += MathHelpers.HSum256_Avx(vacc1);
                            for (; l < halfRem; l++)
                            {
                                int pos = b * QK_K + j + l;
                                sum += input[pos] * ((sc0 * (qs[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper));
                            }
                        }
                        else
                        {
                            for (int l = 0; l < halfRem; l++)
                            {
                                int pos = b * QK_K + j + l;
                                sum += input[pos] * ((sc0 * (qs[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper));
                            }
                        }
                        
                        int half2End = Math.Min(32, remaining - 32);
                        if (half2End > 0 && half2End >= 8)
                        {
                            var vacc2 = Vector256<float>.Zero;
                            int l = 0;
                            for (; l <= half2End - 8; l += 8)
                            {
                                float v0 = qs[qIdx + l] >> 4;
                                float v1 = qs[qIdx + l + 1] >> 4;
                                float v2 = qs[qIdx + l + 2] >> 4;
                                float v3 = qs[qIdx + l + 3] >> 4;
                                float v4 = qs[qIdx + l + 4] >> 4;
                                float v5 = qs[qIdx + l + 5] >> 4;
                                float v6 = qs[qIdx + l + 6] >> 4;
                                float v7 = qs[qIdx + l + 7] >> 4;
                                var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                                var vi = Vector256.LoadUnsafe(ref pIn[j + 32 + l]);
                                var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1_d), vm1_min));
                                vacc2 = Avx.Add(vacc2, res);
                            }
                            sum += MathHelpers.HSum256_Avx(vacc2);
                            for (; l < half2End; l++)
                            {
                                int pos = b * QK_K + j + 32 + l;
                                sum += input[pos] * ((sc1 * (qs[qIdx + l] >> 4) * dSuper) - (m1 * minSuper));
                            }
                        }
                        else
                        {
                            for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                            {
                                int pos = b * QK_K + j + 32 + l;
                                sum += input[pos] * ((sc1 * (qs[qIdx + l] >> 4) * dSuper) - (m1 * minSuper));
                            }
                        }
                idx += 2;
            }
        }
        return (float)sum;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ5K — 5-bit K-quant
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ5K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 176;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            float min    = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qh     = block + 16;
            byte* qs     = block + 48;

            int idx = 0, qIdx = 0;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0 = GetScaleMinK4_Scale_Scalar(idx + 0, scales);
                byte m0  = GetScaleMinK4_Min_Scalar(idx + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(idx + 1, scales);
                byte m1  = GetScaleMinK4_Min_Scalar(idx + 1, scales);

                int remaining = Math.Min(64, inFeatures - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);
                for (int l = 0; l < halfRem; l++)
                {
                    int pos = b * QK_K + j + l;
                    int val = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                    sum += input[pos] * (sc0 * val * d - m0 * min);
                }
                for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                {
                    int pos = b * QK_K + j + 32 + l;
                    int val = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                    sum += input[pos] * (sc1 * val * d - m1 * min);
                }
                qIdx += 32;
                idx += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ5K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 176;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            float min    = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qh     = block + 16;
            byte* qs     = block + 48;

            float* pIn = input + b * QK_K;
            int idx = 0, qIdx = 0;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0 = GetScaleMinK4_Scale_Scalar(idx + 0, scales);
                byte m0  = GetScaleMinK4_Min_Scalar(idx + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(idx + 1, scales);
                byte m1  = GetScaleMinK4_Min_Scalar(idx + 1, scales);
                var vs0 = Vector256.Create((float)sc0);
                var vm0 = Vector256.Create((float)m0);
                var vs1 = Vector256.Create((float)sc1);
                var vm1 = Vector256.Create((float)m1);
                var vd = Vector256.Create(d);
                var vmin = Vector256.Create(min);
                
                var vs0_d = Avx.Multiply(vs0, vd);
                var vm0_min = Avx.Multiply(vm0, vmin);
                var vs1_d = Avx.Multiply(vs1, vd);
                var vm1_min = Avx.Multiply(vm1, vmin);
                
                int remaining = Math.Min(64, inFeatures - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);

                if (halfRem >= 8)
                {
                    var vacc1 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= halfRem - 8; l += 8)
                    {
                        float v0 = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                        float v1 = (qs[qIdx + l + 1] & 0x0F) + ((qh[l + 1] & u1) != 0 ? 16 : 0);
                        float v2 = (qs[qIdx + l + 2] & 0x0F) + ((qh[l + 2] & u1) != 0 ? 16 : 0);
                        float v3 = (qs[qIdx + l + 3] & 0x0F) + ((qh[l + 3] & u1) != 0 ? 16 : 0);
                        float v4 = (qs[qIdx + l + 4] & 0x0F) + ((qh[l + 4] & u1) != 0 ? 16 : 0);
                        float v5 = (qs[qIdx + l + 5] & 0x0F) + ((qh[l + 5] & u1) != 0 ? 16 : 0);
                        float v6 = (qs[qIdx + l + 6] & 0x0F) + ((qh[l + 6] & u1) != 0 ? 16 : 0);
                        float v7 = (qs[qIdx + l + 7] & 0x0F) + ((qh[l + 7] & u1) != 0 ? 16 : 0);
                                var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                                var vi = Vector256.LoadUnsafe(ref pIn[j + l]);
                                var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0_d), vm0_min));
                                vacc1 = Avx.Add(vacc1, res);
                            }
                            sum += MathHelpers.HSum256_Avx(vacc1);
                            for (; l < halfRem; l++)
                            {
                                int pos = b * QK_K + j + l;
                                int val = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                                sum += input[pos] * (sc0 * val * d - m0 * min);
                            }
                        }
                        else
                        {
                            for (int l = 0; l < halfRem; l++)
                            {
                                int pos = b * QK_K + j + l;
                                int val = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                                sum += input[pos] * (sc0 * val * d - m0 * min);
                            }
                        }
                        
                        int half2End = Math.Min(32, remaining - 32);
                        if (half2End > 0 && half2End >= 8)
                        {
                            var vacc2 = Vector256<float>.Zero;
                            int l = 0;
                            for (; l <= half2End - 8; l += 8)
                            {
                                float v0 = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                                float v1 = (qs[qIdx + l + 1] >> 4) + ((qh[l + 1] & u2) != 0 ? 16 : 0);
                                float v2 = (qs[qIdx + l + 2] >> 4) + ((qh[l + 2] & u2) != 0 ? 16 : 0);
                                float v3 = (qs[qIdx + l + 3] >> 4) + ((qh[l + 3] & u2) != 0 ? 16 : 0);
                                float v4 = (qs[qIdx + l + 4] >> 4) + ((qh[l + 4] & u2) != 0 ? 16 : 0);
                                float v5 = (qs[qIdx + l + 5] >> 4) + ((qh[l + 5] & u2) != 0 ? 16 : 0);
                                float v6 = (qs[qIdx + l + 6] >> 4) + ((qh[l + 6] & u2) != 0 ? 16 : 0);
                                float v7 = (qs[qIdx + l + 7] >> 4) + ((qh[l + 7] & u2) != 0 ? 16 : 0);
                                var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                                var vi = Vector256.LoadUnsafe(ref pIn[j + 32 + l]);
                                var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1_d), vm1_min));
                                vacc2 = Avx.Add(vacc2, res);
                            }
                            sum += MathHelpers.HSum256_Avx(vacc2);
                            for (; l < half2End; l++)
                            {
                                int pos = b * QK_K + j + 32 + l;
                                int val = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                                sum += input[pos] * (sc1 * val * d - m1 * min);
                            }
                        }
                        else
                        {
                            for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                            {
                                int pos = b * QK_K + j + 32 + l;
                                int val = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                                sum += input[pos] * (sc1 * val * d - m1 * min);
                            }
                        }
                qIdx += 32;
                idx += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ5K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 176;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            float min    = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qh     = block + 16;
            byte* qs     = block + 48;

            float* pIn = input + b * QK_K;
            int idx = 0, qIdx = 0;
            byte u1 = 1, u2 = 2;
            for (int j = 0; j < QK_K; j += 64)
            {
                byte sc0 = GetScaleMinK4_Scale_Scalar(idx + 0, scales);
                byte m0  = GetScaleMinK4_Min_Scalar(idx + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(idx + 1, scales);
                byte m1  = GetScaleMinK4_Min_Scalar(idx + 1, scales);
                float d1 = d * sc0; float m1v = min * m0;
                float d2 = d * sc1; float m2v = min * m1;
                var vd1 = Vector256.Create(d1);
                var vm1 = Vector256.Create(m1v);
                var vd2 = Vector256.Create(d2);
                var vm2 = Vector256.Create(m2v);

                int remaining = Math.Min(64, inFeatures - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);

                if (halfRem >= 8)
                {
                    var vacc1 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= halfRem - 8; l += 8)
                    {
                        float v0 = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                        float v1 = (qs[qIdx + l + 1] & 0x0F) + ((qh[l + 1] & u1) != 0 ? 16 : 0);
                        float v2 = (qs[qIdx + l + 2] & 0x0F) + ((qh[l + 2] & u1) != 0 ? 16 : 0);
                        float v3 = (qs[qIdx + l + 3] & 0x0F) + ((qh[l + 3] & u1) != 0 ? 16 : 0);
                        float v4 = (qs[qIdx + l + 4] & 0x0F) + ((qh[l + 4] & u1) != 0 ? 16 : 0);
                        float v5 = (qs[qIdx + l + 5] & 0x0F) + ((qh[l + 5] & u1) != 0 ? 16 : 0);
                        float v6 = (qs[qIdx + l + 6] & 0x0F) + ((qh[l + 6] & u1) != 0 ? 16 : 0);
                        float v7 = (qs[qIdx + l + 7] & 0x0F) + ((qh[l + 7] & u1) != 0 ? 16 : 0);
                        var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                        var vi = Vector256.LoadUnsafe(ref pIn[j + l]);
                        vacc1 = Fma.MultiplyAdd(vi, Avx.Multiply(vv, vd1), vacc1);
                        vacc1 = Avx.Subtract(vacc1, Avx.Multiply(vi, vm1));
                    }
                    sum += MathHelpers.HSum256_Avx(vacc1);
                    for (; l < halfRem; l++)
                    {
                        int pos = b * QK_K + j + l;
                        int val = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                        sum += input[pos] * (d1 * val - m1v);
                    }
                }
                else
                {
                    for (int l = 0; l < halfRem; l++)
                    {
                        int pos = b * QK_K + j + l;
                        int val = (qs[qIdx + l] & 0x0F) + ((qh[l] & u1) != 0 ? 16 : 0);
                        sum += input[pos] * (d1 * val - m1v);
                    }
                }

                int half2End = Math.Min(32, remaining - 32);
                if (half2End > 0 && half2End >= 8)
                {
                    var vacc2 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= half2End - 8; l += 8)
                    {
                        float v0 = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                        float v1 = (qs[qIdx + l + 1] >> 4) + ((qh[l + 1] & u2) != 0 ? 16 : 0);
                        float v2 = (qs[qIdx + l + 2] >> 4) + ((qh[l + 2] & u2) != 0 ? 16 : 0);
                        float v3 = (qs[qIdx + l + 3] >> 4) + ((qh[l + 3] & u2) != 0 ? 16 : 0);
                        float v4 = (qs[qIdx + l + 4] >> 4) + ((qh[l + 4] & u2) != 0 ? 16 : 0);
                        float v5 = (qs[qIdx + l + 5] >> 4) + ((qh[l + 5] & u2) != 0 ? 16 : 0);
                        float v6 = (qs[qIdx + l + 6] >> 4) + ((qh[l + 6] & u2) != 0 ? 16 : 0);
                        float v7 = (qs[qIdx + l + 7] >> 4) + ((qh[l + 7] & u2) != 0 ? 16 : 0);
                        var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                        var vi = Vector256.LoadUnsafe(ref pIn[j + 32 + l]);
                        vacc2 = Fma.MultiplyAdd(vi, Avx.Multiply(vv, vd2), vacc2);
                        vacc2 = Avx.Subtract(vacc2, Avx.Multiply(vi, vm2));
                    }
                    sum += MathHelpers.HSum256_Avx(vacc2);
                    for (; l < half2End; l++)
                    {
                        int pos = b * QK_K + j + 32 + l;
                        int val = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                        sum += input[pos] * (d2 * val - m2v);
                    }
                }
                else
                {
                    for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                    {
                        int pos = b * QK_K + j + 32 + l;
                        int val = (qs[qIdx + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0);
                        sum += input[pos] * (d2 * val - m2v);
                    }
                }
                qIdx += 32;
                idx += 2;
                u1 <<= 2; u2 <<= 2;
            }
        }
        return (float)sum;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ6K — 6-bit K-quant
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ6K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* ql     = block;
            byte* qh     = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d      = HalfToFloat_Scalar(*(ushort*)(block + 208));

            int valid = Math.Min(QK_K, inFeatures - b * QK_K);
            for (int nOff = 0; nOff < valid; nOff += 128)
            {
                byte* pql  = ql + (nOff == 0 ? 0 : 64);
                byte* pqh  = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                int halfRem = Math.Min(128, valid - nOff);
                for (int l = 0; l < 32 && l < halfRem; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = b * QK_K + nOff + l;
                    int i2 = b * QK_K + nOff + l + 32;

                    if (i2 >= b * QK_K + valid)
                    {
                        if (i1 < b * QK_K + valid)
                            sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                        break;
                    }

                    int i3 = b * QK_K + nOff + l + 64;
                    int i4 = b * QK_K + nOff + l + 96;

                    sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                    sum += input[i2] * (d * psc[is_ + 2] * (q2v - 32));
                    sum += input[i3] * (d * psc[is_ + 4] * (q3v - 32));
                    sum += input[i4] * (d * psc[is_ + 6] * (q4v - 32));
                }
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ6K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* ql     = block;
            byte* qh     = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d      = HalfToFloat_Scalar(*(ushort*)(block + 208));

            int valid = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;
            for (int nOff = 0; nOff < valid; nOff += 128)
            {
                byte* pql  = ql + (nOff == 0 ? 0 : 64);
                byte* pqh  = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                int halfRem = Math.Min(128, valid - nOff);
                for (int l = 0; l < 32 && l < halfRem; l++)
                {
                    int is_   = l / 16;
                    int q1v   = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v   = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v   = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v   = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = nOff + l;
                    int i2 = nOff + l + 32;

                    if (i2 >= valid)
                    {
                        if (i1 < valid)
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
        return (float)sum;
    }

    internal static unsafe float VecDotQ6K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* ql     = block;
            byte* qh     = block + 128;
            sbyte* scales = (sbyte*)(block + 192);
            float d      = HalfToFloat_Scalar(*(ushort*)(block + 208));

            int valid = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn = input + b * QK_K;
            for (int nOff = 0; nOff < valid; nOff += 128)
            {
                byte* pql  = ql + (nOff == 0 ? 0 : 64);
                byte* pqh  = qh + (nOff == 0 ? 0 : 32);
                sbyte* psc = scales + (nOff == 0 ? 0 : 8);

                int halfRem = Math.Min(128, valid - nOff);
                for (int l = 0; l < 32 && l < halfRem; l++)
                {
                    int is_   = l / 16;
                    int q1v   = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v   = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v   = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v   = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = nOff + l;
                    int i2 = nOff + l + 32;

                    if (i2 >= valid)
                    {
                        if (i1 < valid)
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
        return (float)sum;
    }
}
