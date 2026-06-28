using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    internal const int QK_K = 256;
    internal const int QK = 32;


    // HalfToFloat — FP16 → FP32    

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HalfToFloat_F16C(ushort half)
    {
        return (float)BitConverter.UInt16BitsToHalf(half);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float HalfToFloat_Scalar(ushort half)
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


    // FloatToHalf — FP32 → FP16

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort FloatToHalf_F16C(float f) => BitConverter.HalfToUInt16Bits((Half)f);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe ushort FloatToHalf_Scalar(float f)
    {
        uint bits = *(uint*)&f;
        uint sign = (bits >> 16) & 0x8000;
        int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint mant = bits & 0x007FFFFF;

        if (exp <= 0)
        {
            if (exp < -10) return (ushort)sign;
            mant = (mant | 0x00800000) >> (1 - exp);
            return (ushort)(sign | (mant >> 13));
        }

        if (exp >= 31)
        {
            if (exp > 31) return (ushort)(sign | 0x7C00 | (mant >> 13));
            return (ushort)(sign | 0x7C00 | (mant != 0 ? 0x200u : 0));
        }

        uint eBits = (uint)(exp << 10);
        uint mMant = mant >> 13;
        return (ushort)(sign | eBits | mMant);
    }

    // GetScaleMinK4 — 6-bit K-quant scale/min unpacking

    public static unsafe byte GetScaleMinK4_Scale_Scalar(int j, byte* scales)
    {
        if (j < 4)
            return (byte)(scales[j] & 0x3F);
        return (byte)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
    }

    public static unsafe byte GetScaleMinK4_Min_Scalar(int j, byte* scales)
    {
        if (j < 4)
            return (byte)(scales[j + 4] & 0x3F);
        return (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }

    // VecDotQ3K — 3-bit K-quant

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
                sum += MathHelpers.HSum256_Avx(Fma.MultiplyAdd(vi, vv, Vector256<float>.Zero));
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

    // VecDotQ4K — 4-bit K-quant

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
            float dSuper = HalfToFloat_Scalar(*(ushort*)block);
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 16;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int j = curBlockStart; j < blockEnd; j += 64)
            {
                int scOff = (j / 64) * 2;
                byte sc0 = GetScaleMinK4_Scale_Scalar(scOff + 0, scales);
                byte m0 = GetScaleMinK4_Min_Scalar(scOff + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(scOff + 1, scales);
                byte m1 = GetScaleMinK4_Min_Scalar(scOff + 1, scales);
                float d1 = dSuper * sc0;
                float m1v = minSuper * m0;
                float d2 = dSuper * sc1;
                float m2v = minSuper * m1;

                int qIdx = (j / 64) * 32;
                int remaining = Math.Min(64, inFeatures + colBlockStart - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);
                for (int l = 0; l < halfRem; l++)
                {
                    int pos = b * QK_K + j + l - colBlockStart;
                    sum += input[pos] * (d1 * (qs[qIdx + l] & 0x0F) - m1v);
                }
                for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                {
                    int pos = b * QK_K + j + 32 + l - colBlockStart;
                    sum += input[pos] * (d2 * (qs[qIdx + l] >> 4) - m2v);
                }
            }
        }
        return (float)sum;
    }

    public static unsafe float VecDotQ4K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float dSuper = HalfToFloat_F16C(*(ushort*)block);
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 16;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int j = curBlockStart; j < blockEnd; j += 64)
            {
                int scOff = (j / 64) * 2;
                byte sc0 = GetScaleMinK4_Scale_Scalar(scOff + 0, scales);
                byte m0 = GetScaleMinK4_Min_Scalar(scOff + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(scOff + 1, scales);
                byte m1 = GetScaleMinK4_Min_Scalar(scOff + 1, scales);
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
                int remaining = Math.Min(64, inFeatures + colBlockStart - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);

                if (halfRem >= 8)
                {
                    var vacc1 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= Math.Min(halfRem - 8, 112 - qIdx); l += 8)
                    {
                        var packed = Sse2.LoadVector128((byte*)(qs + qIdx + l));
                        var vv = Avx.ConvertToVector256Single(Avx2.And(Avx2.ConvertToVector256Int32(packed), Vector256.Create(0x0F)));
                        var vi = Vector256.LoadUnsafe(ref pIn[j + l]);
                        var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0_d), vm0_min));
                        vacc1 = Avx.Add(vacc1, res);
                    }
                    sum += MathHelpers.HSum256_Avx(vacc1);
                    for (; l < halfRem; l++)
                    {
                        int pos = b * QK_K + j + l - colBlockStart;
                        sum += input[pos] * ((sc0 * (qs[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper));
                    }
                }
                else
                {
                    for (int l = 0; l < halfRem; l++)
                    {
                        int pos = b * QK_K + j + l - colBlockStart;
                        sum += input[pos] * ((sc0 * (qs[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper));
                    }
                }

                int half2End = Math.Min(32, remaining - 32);
                if (half2End > 0 && half2End >= 8)
                {
                    var vacc2 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= Math.Min(half2End - 8, 112 - qIdx); l += 8)
                    {
                        var packed = Sse2.LoadVector128((byte*)(qs + qIdx + l));
                        var vv = Avx.ConvertToVector256Single(Avx2.ShiftRightLogical(Avx2.ConvertToVector256Int32(packed), 4));
                        var vi = Vector256.LoadUnsafe(ref pIn[j + 32 + l]);
                        var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1_d), vm1_min));
                        vacc2 = Avx.Add(vacc2, res);
                    }
                    sum += MathHelpers.HSum256_Avx(vacc2);
                    for (; l < half2End; l++)
                    {
                        int pos = b * QK_K + j + 32 + l - colBlockStart;
                        sum += input[pos] * ((sc1 * (qs[qIdx + l] >> 4) * dSuper) - (m1 * minSuper));
                    }
                }
                else
                {
                    for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                    {
                        int pos = b * QK_K + j + 32 + l - colBlockStart;
                        sum += input[pos] * ((sc1 * (qs[qIdx + l] >> 4) * dSuper) - (m1 * minSuper));
                    }
                }
            }
        }
        return (float)sum;
    }

    public static unsafe float VecDotQ4K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float dSuper = HalfToFloat_F16C(*(ushort*)block);
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 16;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int j = curBlockStart; j < blockEnd; j += 64)
            {
                int scOff = (j / 64) * 2;
                byte sc0 = GetScaleMinK4_Scale_Scalar(scOff + 0, scales);
                byte m0 = GetScaleMinK4_Min_Scalar(scOff + 0, scales);
                byte sc1 = GetScaleMinK4_Scale_Scalar(scOff + 1, scales);
                byte m1 = GetScaleMinK4_Min_Scalar(scOff + 1, scales);
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
                int remaining = Math.Min(64, inFeatures + colBlockStart - b * QK_K - j);
                int halfRem = Math.Min(32, remaining);

                if (halfRem >= 8)
                {
                    var vacc1 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= Math.Min(halfRem - 8, 112 - qIdx); l += 8)
                    {
                        var packed = Sse2.LoadVector128((byte*)(qs + qIdx + l));
                        var vv = Avx.ConvertToVector256Single(Avx2.And(Avx2.ConvertToVector256Int32(packed), Vector256.Create(0x0F)));
                        var vi = Vector256.LoadUnsafe(ref pIn[j + l]);
                        var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0_d), vm0_min));
                        vacc1 = Avx.Add(vacc1, res);
                    }
                    sum += MathHelpers.HSum256_Avx(vacc1);
                    for (; l < halfRem; l++)
                    {
                        int pos = b * QK_K + j + l - colBlockStart;
                        sum += input[pos] * ((sc0 * (qs[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper));
                    }
                }
                else
                {
                    for (int l = 0; l < halfRem; l++)
                    {
                        int pos = b * QK_K + j + l - colBlockStart;
                        sum += input[pos] * ((sc0 * (qs[qIdx + l] & 0x0F) * dSuper) - (m0 * minSuper));
                    }
                }

                int half2End = Math.Min(32, remaining - 32);
                if (half2End > 0 && half2End >= 8)
                {
                    var vacc2 = Vector256<float>.Zero;
                    int l = 0;
                    for (; l <= Math.Min(half2End - 8, 112 - qIdx); l += 8)
                    {
                        var packed = Sse2.LoadVector128((byte*)(qs + qIdx + l));
                        var vv = Avx.ConvertToVector256Single(Avx2.ShiftRightLogical(Avx2.ConvertToVector256Int32(packed), 4));
                        var vi = Vector256.LoadUnsafe(ref pIn[j + 32 + l]);
                        var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1_d), vm1_min));
                        vacc2 = Avx.Add(vacc2, res);
                    }
                    sum += MathHelpers.HSum256_Avx(vacc2);
                    for (; l < half2End; l++)
                    {
                        int pos = b * QK_K + j + 32 + l - colBlockStart;
                        sum += input[pos] * ((sc1 * (qs[qIdx + l] >> 4) * dSuper) - (m1 * minSuper));
                    }
                }
                else
                {
                    for (int l = 0; l < 32 && (32 + l) < remaining; l++)
                    {
                        int pos = b * QK_K + j + 32 + l - colBlockStart;
                        sum += input[pos] * ((sc1 * (qs[qIdx + l] >> 4) * dSuper) - (m1 * minSuper));
                    }
                }
            }
        }
        return (float)sum;
    }

    // VecDotQ5K — 5-bit K-quant
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
                int subIdx = i / 32;
                byte sc = GetScaleMinK4_Scale_Scalar(subIdx, scales);
                byte m = GetScaleMinK4_Min_Scalar(subIdx, scales);

                int qsByte = (i / 64) * 32 + (i % 32);
                int qsShift = ((i % 64) / 32) * 4;
                int q4 = (qs[qsByte] >> qsShift) & 0x0F;

                int qhBit = (qh[i % 32] >> ((i / 64) * 2 + ((i % 64) / 32))) & 1;
                int q5 = q4 | (qhBit << 4);

                sum += input[b * QK_K + i - colBlockStart] * (sc * q5 * d - m * min);
            }
        }
        return (float)sum;
    }

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
            int i = curBlockStart;
            for (; i <= blockEnd - 8; i += 8)
            {
                int subIdx = i / 32;
                float sc = GetScaleMinK4_Scale_Scalar(subIdx, scales);
                float m = GetScaleMinK4_Min_Scalar(subIdx, scales);
                var vd = Vector256.Create(d * sc);
                var vm = Vector256.Create(m * min);

                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int qsByte = (idx / 64) * 32 + (idx % 32);
                    int qsShift = ((idx % 64) / 32) * 4;
                    int q4 = (qs[qsByte] >> qsShift) & 0x0F;
                    int qhBit = (qh[idx % 32] >> ((idx / 64) * 2 + ((idx % 64) / 32))) & 1;
                    vvBuf[sub] = q4 | (qhBit << 4);
                }
                var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vd), vm));
                sum += MathHelpers.HSum256_Avx(res);
            }
            for (; i < blockEnd; i++)
            {
                int subIdx = i / 32;
                byte sc = GetScaleMinK4_Scale_Scalar(subIdx, scales);
                byte m = GetScaleMinK4_Min_Scalar(subIdx, scales);

                int qsByte = (i / 64) * 32 + (i % 32);
                int qsShift = ((i % 64) / 32) * 4;
                int q4 = (qs[qsByte] >> qsShift) & 0x0F;

                int qhBit = (qh[i % 32] >> ((i / 64) * 2 + ((i % 64) / 32))) & 1;
                int q5 = q4 | (qhBit << 4);

                sum += pIn[i] * (sc * q5 * d - m * min);
            }
        }
        return (float)sum;
    }

    public static unsafe float VecDotQ5K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
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
            int i = curBlockStart;
            for (; i <= blockEnd - 8; i += 8)
            {
                int subIdx = i / 32;
                float sc = GetScaleMinK4_Scale_Scalar(subIdx, scales);
                float m = GetScaleMinK4_Min_Scalar(subIdx, scales);
                var vd = Vector256.Create(d * sc);
                var vm = Vector256.Create(m * min);

                for (int sub = 0; sub < 8; sub++)
                {
                    int idx = i + sub;
                    int qsByte = (idx / 64) * 32 + (idx % 32);
                    int qsShift = ((idx % 64) / 32) * 4;
                    int q4 = (qs[qsByte] >> qsShift) & 0x0F;
                    int qhBit = (qh[idx % 32] >> ((idx / 64) * 2 + ((idx % 64) / 32))) & 1;
                    vvBuf[sub] = q4 | (qhBit << 4);
                }
                var vv = Vector256.LoadUnsafe(ref vvBuf[0]);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.Subtract(Avx.Multiply(vv, vd), vm);
                var res = Avx.Multiply(vi, vw);
                sum += MathHelpers.HSum256_Avx(res);
            }
            for (; i < blockEnd; i++)
            {
                int subIdx = i / 32;
                byte sc = GetScaleMinK4_Scale_Scalar(subIdx, scales);
                byte m = GetScaleMinK4_Min_Scalar(subIdx, scales);

                int qsByte = (i / 64) * 32 + (i % 32);
                int qsShift = ((i % 64) / 32) * 4;
                int q4 = (qs[qsByte] >> qsShift) & 0x0F;

                int qhBit = (qh[i % 32] >> ((i / 64) * 2 + ((i % 64) / 32))) & 1;
                int q5 = q4 | (qhBit << 4);

                sum += pIn[i] * (sc * q5 * d - m * min);
            }
        }
        return (float)sum;
    }

    // QuantizedMatMulQ6K — fused matmul for q6_k weights

    public static unsafe void QuantizedMatMulQ6K_Scalar(
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

    public static unsafe float VecDotQ6K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
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
                for (int l = 0; l < 32 && l < halfRem; l++)
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
        return (float)sum;
    }

    public static unsafe float VecDotQ6K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
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
                for (int l = 0; l < 32 && l < halfRem; l++)
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
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulQ6K_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            System.Threading.Tasks.Parallel.For(0, N, col =>
            {
                output[col] = VecDotQ6K_AVX2(input, rawWeights, col, K);
            });
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

    public static unsafe void QuantizedMatMulQ6K_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            System.Threading.Tasks.Parallel.For(0, N, col =>
            {
                output[col] = VecDotQ6K_FMA(input, rawWeights, col, K);
            });
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
