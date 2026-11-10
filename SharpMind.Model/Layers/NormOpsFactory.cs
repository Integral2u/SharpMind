using JigSawDotNet;
using SharpMind.Core;
using System.Collections.Concurrent;

namespace SharpMind.Model.Layers;

public static class NormOpsFactory
{
    private static readonly ConcurrentDictionary<int, NormOps> _opsCache = [];

    public static NormOps Create(SharpMindConfig config) => Create(config.ToJigSawMapping());

    /// <summary>
    /// Kernels for <paramref name="config"/> with <paramref name="overrides"/> applied on top. A
    /// caller's mapping may hold only the keys it cares about (quantization presets, say), so the
    /// norm key is always seeded from the config first.
    /// </summary>
    public static NormOps Create(SharpMindConfig config, Dictionary<string, string>? overrides)
        => Create(config.ToJigSawMapping(overrides));

    public static NormOps Create(Dictionary<string, string> mapping)
        => _opsCache.GetOrAdd(MappingHash.Compute(mapping), _ => Assembler.CreateInstance<NormOps>(mapping));
}
