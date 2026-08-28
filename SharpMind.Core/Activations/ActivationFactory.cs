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

    // Fast last-hit slot: skips the dictionary probe when the same mapping hash is
    // resolved repeatedly. The slot is READ as a pair - "does the hash match? then take
    // the ops" - so it must also be WRITTEN as a pair. Holding it as two fields cannot do
    // that however they are marked: volatile orders each field on its own, it does not
    // publish both at once, so two threads resolving different mappings can interleave
    // into a stored hash from one and stored ops from the other. One immutable record
    // behind one reference write makes the pair indivisible.
    private sealed record LastHit(int Hash, ActivationOps Ops);
    private static LastHit? _lastHit;

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

        if (Volatile.Read(ref _lastHit) is { } hit && hit.Hash == hash)
            return hit.Ops;

        var ops = _opsCache.GetOrAdd(hash, _ => Assembler.CreateInstance<ActivationOps>(mappings));

        Volatile.Write(ref _lastHit, new LastHit(hash, ops));
        return ops;
    }
}
