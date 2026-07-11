using JigSawDotNet;
using System.Collections.Concurrent;

namespace SharpMind.Core.Quantization;

public static class QuantizationFactory
{
    private static readonly ConcurrentDictionary<int, QuantizationOps> _opsCache = [];

    public static QuantizationOps Create(HardwareTier hw = HardwareTier.Auto)
    {
        if (hw == HardwareTier.Auto)
            hw = HardwareTierHelpers.DetectBestTier();
        var mapping = new MappingBuilder(hw).ApplyQuantPreset().Build();
        return Create(mapping);
    }

    public static QuantizationOps Create(Dictionary<string, string> mappings) =>
        _opsCache.GetOrAdd(MappingHash.Compute(mappings), (h) =>
        {
            return Assembler.CreateInstance<QuantizationOps>(mappings);
        });
}
