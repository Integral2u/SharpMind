using System.Collections.Concurrent;
using JigSawDotNet;

namespace SharpMind.Core.Training;

/// <summary>
/// Assembles a <see cref="GradientMapping"/> instance from a <see cref="SharpMindConfig"/>.
/// Hardware tier is resolved once here — no runtime checks inside any kernel.
///
/// Usage:
/// <code>
/// var mapping = GradientMappingFactory.Create(SharpMindConfig.Llama);
/// var dInput  = mapping.Linear(dOutput, input, weight, bias);
/// </code>
/// </summary>
public static class GradientMappingFactory
{
    private static readonly ConcurrentDictionary<int, GradientMapping> _mappingCache = [];

    /// <summary>
    /// Assembles and returns a <see cref="GradientMapping"/> for the given config.
    /// Hardware tier is taken directly from <see cref="SharpMindConfig.ResolvedHardware"/>.
    /// </summary>
    public static GradientMapping Create(SharpMindConfig config)
        => Create(config.ToJigSawMapping());

    /// <summary>
    /// Assembles a <see cref="GradientMapping"/> from the given config, applying
    /// per-key overrides (e.g. GPU-pinned kernels) on top of the resolved mapping.
    /// </summary>
    public static GradientMapping Create(SharpMindConfig config, Dictionary<string, string>? overrides)
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

    /// <summary>Assembles a <see cref="GradientMapping"/> from an explicit mapping.</summary>
    public static GradientMapping Create(Dictionary<string, string> mappings)
        => _mappingCache.GetOrAdd(MappingHash.Compute(mappings),
            _ => Assembler.CreateInstance<GradientMapping>(mappings));
}
