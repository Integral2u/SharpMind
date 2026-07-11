using JigSawDotNet;
using SharpMind.Core;
using System.Collections.Concurrent;

namespace SharpMind.Model.Layers;

public static class NormOpsFactory
{
    private static readonly ConcurrentDictionary<int, NormOps> _opsCache = [];

    public static NormOps Create(SharpMindConfig config)
        => _opsCache.GetOrAdd(MappingHash.Compute(config.ToJigSawMapping()),
            _ => Assembler.CreateInstance<NormOps>(config.ToJigSawMapping()));

    public static NormOps SetDefault(SharpMindConfig config)
    {
        var ops = Create(config);
        NormOps.SetDefault(ops);
        return ops;
    }
}
