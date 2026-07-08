using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    internal const int QK_K = 256;
    internal const int QK = 32;

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
}
