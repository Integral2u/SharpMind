using JigSawDotNet;
using System.Collections.Concurrent;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static class QuantizationFactory
{
    private static readonly ConcurrentDictionary<HardwareTier, QuantizationOps> _tierCache = [];

    public static QuantizationOps Create(HardwareTier hw = HardwareTier.Auto)
    {
        if (hw == HardwareTier.Auto)
            hw = DetectBestTier();

        return _tierCache.GetOrAdd(hw, static tier =>
        {
            var config = new QuantizationConfig { Hardware = tier };
            return Assembler.CreateInstance<QuantizationOps>(config.ToJigSawMapping());
        });
    }

    public static QuantizationOps Create(Dictionary<string, string> mappings) =>
        Assembler.CreateInstance<QuantizationOps>(mappings);

    private static HardwareTier DetectBestTier()
    {
        if (Fma.IsSupported)  return HardwareTier.FMA;
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (Sse.IsSupported)  return HardwareTier.SSE;
        return HardwareTier.Scalar;
    }
}
