using JigSawDotNet;
using SharpMind.Core.Activations;
using System.Collections.Concurrent;

namespace SharpMind.Core.Ops;

/// <summary>
/// Creates <see cref="TensorOps"/> instances wired by JigSawDotNet.
/// </summary>
public static class TensorOpsFactory
{
    private static readonly ConcurrentDictionary<int, TensorOps> _opsCache = [];
    /// <summary>
    /// Assembles and returns a <see cref="TensorOps"/> for the given config.
    /// </summary>
    public static TensorOps Create(SharpMindConfig config)
        => Create(config.ToJigSawMapping());
    public static TensorOps Create(SharpMindConfig config, Dictionary<string, string>? overrides)
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
    public static TensorOps Create(Dictionary<string, string> mappings) =>
        _opsCache.GetOrAdd(mappings.GetHashCode(), (h) =>
        {
            return Assembler.CreateInstance<TensorOps>(mappings);
        });
    /*
    /// <summary>
    /// Assembles a <see cref="TensorOps"/>, sets it as <see cref="TensorOps.Default"/>,
    /// and returns it. Call once at application startup.
    /// </summary>
    public static TensorOps SetDefault(SharpMindConfig config)
    {
        var ops = Create(config);
        TensorOps.SetDefault(ops);
        return ops;
    }*/
}
