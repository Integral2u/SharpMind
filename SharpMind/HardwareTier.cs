using System.Runtime.Intrinsics.X86;

namespace SharpMind;

public enum HardwareTier   { Auto, FMA, AVX2, SSE, Scalar }

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