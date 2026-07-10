using JigSawDotNet;
using SharpMind.Core.Ops;
using System.Collections.Concurrent;
using System.Runtime.Intrinsics.X86;
using System.Timers;

namespace SharpMind.Core.Quantization;

public static class QuantizationFactory
{
    private static readonly ConcurrentDictionary<int, QuantizationOps> _opsCache = [];

    public static QuantizationOps Create(HardwareTier hw = HardwareTier.Auto)
    {
        if (hw == HardwareTier.Auto)
            hw = HardwareTierHelpers.DetectBestTier();
        var config = new QuantizationConfig { Hardware = hw };
        return Create(config.ToJigSawMapping());
    }
    public static QuantizationOps Create(QuantizationConfig config)
       => Create(config.ToJigSawMapping());
    public static QuantizationOps Create(QuantizationConfig config, Dictionary<string, string>? overrides)
    {
        var cfg = config.ToJigSawMapping();
        if (overrides != null)
        {
            foreach (var m in overrides)
            {
                if (cfg.TryGetValue(m.Key, out string? value)) cfg[m.Key] = value;
                else cfg.Add(m.Key, m.Value);
            }
        }
        return Create(cfg);
    }
    public static QuantizationOps Create(Dictionary<string, string> mappings) =>
        _opsCache.GetOrAdd(mappings.GetHashCode(), (h) =>
        {
            return Assembler.CreateInstance<QuantizationOps>(mappings);
        });

   /* public static QuantizationOps Create(HardwareTier hw = HardwareTier.Auto)
    {
        if (hw == HardwareTier.Auto)
            hw = HardwareTierHelpers.DetectBestTier();

        return _tierCache.GetOrAdd(hw, static tier =>
        {
            var config = new QuantizationConfig { Hardware = tier };
            return Assembler.CreateInstance<QuantizationOps>(config.ToJigSawMapping());
        });
    }

    public static QuantizationOps Create(Dictionary<string, string> mappings) =>
        Assembler.CreateInstance<QuantizationOps>(mappings);*/

}
