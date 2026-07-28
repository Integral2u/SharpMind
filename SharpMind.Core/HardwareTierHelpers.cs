using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core;

public static class HardwareTierHelpers
{
    public static HardwareTier DetectBestTier()
    {
        if (Fma.IsSupported) return HardwareTier.FMA;
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (Sse.IsSupported) return HardwareTier.SSE;
        return HardwareTier.Scalar;
    }
}