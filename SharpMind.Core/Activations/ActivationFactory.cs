using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
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
        _opsCache.GetOrAdd(mappings.GetHashCode(), (h) =>
            {
                return Assembler.CreateInstance<ActivationOps>(mappings);
            });
    /// <summary>
    /// Assembles both hardware variants available on this machine, benchmarks them
    /// directly (bypassing reflection — <see cref="System.Span{T}"/> cannot be boxed),
    /// and returns whichever is fastest.
    ///
    /// Act type and gate type are always pinned from <paramref name="config"/>;
    /// only the hardware tier is contested.
    /// </summary>
    /// <param name="config">Activation profile. Hardware field is ignored — both
    /// available tiers are tried.</param>
    /// <param name="benchVectorSize">Length of the test vector. Should be
    /// representative of the model's hidden dimension.</param>
    /// <param name="warmup">Iterations before timing begins.</param>
    /// <param name="iterations">Timed iterations per candidate.</param>
    public static ActivationOps CreateForSystem(
        SharpMindConfig config,
        int benchVectorSize = 4_096,
        int warmup = 200,
        int iterations = 2_000)
    {
        // Scalar is always available. AVX2 only if the CPU supports it.
        var scalarInstance = Create(config with { Hardware = HardwareTier.Scalar });

        if (!Avx2.IsSupported)
            return scalarInstance;

        var avx2Instance = Create(config with { Hardware = HardwareTier.AVX2 });

        return BenchmarkRMSNorm(avx2Instance, scalarInstance, benchVectorSize, warmup, iterations)
            ? avx2Instance
            : scalarInstance;
    }

    /// <summary>
    /// Returns true if <paramref name="avx2"/> is faster than <paramref name="scalar"/>
    /// on this machine for RMSNorm — the only kernel where AVX2 provides a real gain
    /// (pure multiply, unlike softmax/GELU/SiLU which are exp/tanh bound).
    /// </summary>
    private static bool BenchmarkRMSNorm(
        ActivationOps avx2,
        ActivationOps scalar,
        int size,
        int warmup,
        int iterations)
    {
        using var src = Tensor<float>.Ones(size);
        using var weight = Tensor<float>.Ones(size);
        using var result = new Tensor<float>(size);

        ReadOnlySpan<float> srcSpan = src.Data;
        ReadOnlySpan<float> weightSpan = weight.Data;
        Span<float> dstSpan = result.Data;
        const float rmsInv = 1f;

        for (int i = 0; i < warmup; i++)
        {
            avx2.ApplyRMSNormRow(srcSpan, weightSpan, dstSpan, rmsInv);
            scalar.ApplyRMSNormRow(srcSpan, weightSpan, dstSpan, rmsInv);
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            avx2.ApplyRMSNormRow(srcSpan, weightSpan, dstSpan, rmsInv);
        long avx2Ticks = sw.ElapsedTicks;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
            scalar.ApplyRMSNormRow(srcSpan, weightSpan, dstSpan, rmsInv);
        long scalarTicks = sw.ElapsedTicks;

        return avx2Ticks <= scalarTicks;
    }
}