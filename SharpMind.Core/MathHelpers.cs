using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core;

public static class MathHelpers
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HSum256_Avx(Vector256<float> v)
    {
        var lo = Avx.ExtractVector128(v, 0);
        var hi = Avx.ExtractVector128(v, 1);
        var s = Sse.Add(lo, hi);
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        return Sse.AddScalar(s, Sse.Shuffle(s, s, 1)).ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HSum256_Sse3(Vector256<float> v)
    {
        var lo = v.GetLower();
        var hi = v.GetUpper();
        var s = Sse.Add(lo, hi);
        var s2 = Sse3.HorizontalAdd(s, s);
        return s2.GetElement(0) + s2.GetElement(1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HSum128_Sse(Vector128<float> v)
    {
        var s = Sse.Add(v, Sse.MoveHighToLow(v, v));
        return Sse.AddScalar(s, Sse.Shuffle(s, s, 1)).ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HSum256_Scalar(Vector256<float> v)
    {
        return v.GetElement(0) + v.GetElement(1) + v.GetElement(2) + v.GetElement(3)
             + v.GetElement(4) + v.GetElement(5) + v.GetElement(6) + v.GetElement(7);
    }
}
