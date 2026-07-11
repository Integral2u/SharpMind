using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using JigSawDotNet;
using SharpMind.Core.Tensors;

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
/// </summary>
public static class ActivationFactory
{
    private static readonly ConcurrentDictionary<int, ActivationOps> _opsCache = [];
    /// <summary>
    /// Assembles and returns an <see cref="ActivationOps"/> for the given config.
    /// Hardware tier is taken directly from <see cref="SharpMindConfig.ResolvedHardware"/>.
    /// </summary>
    public static ActivationOps Create(SharpMindConfig config)
        => Create(config.ToJigSawMapping());
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
    public static ActivationOps Create(Dictionary<string, string> mappings) => 
        _opsCache.GetOrAdd(MappingHash.Compute(mappings), (h) =>
            {
                return Assembler.CreateInstance<ActivationOps>(mappings);
            });    
}