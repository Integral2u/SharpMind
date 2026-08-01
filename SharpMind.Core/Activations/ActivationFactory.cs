using System.Collections.Concurrent;
using JigSawDotNet;

namespace SharpMind.Core.Activations;

/// <summary>
/// Creates <see cref="ActivationOps"/> instances wired by JigSawDotNet.
///
/// Hardware tier is resolved once at factory time — no runtime checks exist
/// inside any assembled kernel.
///
/// For different LLM architectures pass the matching preset:
/// <code>
///   var gptOps   = ActivationFactory.Create(SharpMindConfig.Gpt);
///   var llamaOps = ActivationFactory.Create(SharpMindConfig.Llama);
/// </code>
/// Both instances coexist — each is its own assembled type cached by JigSaw.
///
/// Lookups are served from three tiers, all keyed so that equal mappings
/// always resolve to the same <see cref="ActivationOps"/> instance:
///   1. a fast single-slot last-hit cache (the common case — the same mapping
///      is resolved repeatedly),
///   2. a type-indexed cache guarded by the config record's structural
///      equality, so repeated <see cref="Create(SharpMindConfig)"/> calls with
///      an equal config skip mapping construction entirely,
///   3. the master dictionary keyed by <see cref="MappingHash.Compute"/> for
///      arbitrary mapping dictionaries.
/// </summary>
public static class ActivationFactory
{
    private static readonly ConcurrentDictionary<int, ActivationOps> _opsCache = [];

    // Fast last-hit slot: skips the dictionary probe when the same mapping
    // hash is resolved repeatedly. Volatile semantics keep the hash/ops pair
    // consistent across concurrent readers and writers.
    private static int _lastHash;
    private static ActivationOps? _lastOps;

    // Type-indexed cache: keyed by the config runtime type so a repeated
    // Create(config) call with an equal SharpMindConfig (a sealed record, so
    // == is structural) resolves without rebuilding the JigSaw mapping. The
    // stored source config guards correctness: a same-type config with a
    // different value falls through to the hash-based path.
    private static readonly ConcurrentDictionary<Type, (SharpMindConfig Source, ActivationOps Ops)> _typeCache = [];

    /// <summary>
    /// Assembles and returns an <see cref="ActivationOps"/> for the given config.
    /// Hardware tier is taken directly from <see cref="SharpMindConfig.ResolvedHardware"/>.
    /// </summary>
    public static ActivationOps Create(SharpMindConfig config)
    {
        Type type = config.GetType();
        if (_typeCache.TryGetValue(type, out var hit) && hit.Source == config)
            return hit.Ops;

        var ops = Create(config.ToJigSawMapping());
        _typeCache[type] = (config, ops);
        return ops;
    }

    public static ActivationOps Create(SharpMindConfig config, Dictionary<string, string>? overrides)
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

    public static ActivationOps Create(Dictionary<string, string> mappings)
    {
        int hash = MappingHash.Compute(mappings);

        if (Volatile.Read(ref _lastHash) == hash && Volatile.Read(ref _lastOps) is { } hit)
            return hit;

        var ops = _opsCache.GetOrAdd(hash, _ => Assembler.CreateInstance<ActivationOps>(mappings));

        Volatile.Write(ref _lastHash, hash);
        Volatile.Write(ref _lastOps, ops);
        return ops;
    }
}
