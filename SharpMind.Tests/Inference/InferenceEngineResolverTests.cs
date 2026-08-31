using SharpMind.Core;
using SharpMind.Core.Plugins;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// The refusal contract, device-independent (no real accelerator, no real model): an
/// accelerator choice is honoured or the launch fails — never a silent CPU fallback.
/// Mirrors the training resolver's contract, which these tests intentionally cross-check.
/// </summary>
public sealed class InferenceEngineResolverTests
{
    private static InferenceEngineContext Ctx => new(null!, null!, MaxCacheLength: 64);

    private sealed record FakeEngine : IInferenceEngine
    {
        public int CachedLength => 0;
        public int MaxCacheLength => 64;
        public bool IsCacheFull => false;
        public string Description => "CPU";
        public ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokenIds, Action<double>? onChunkProgress = null, CancellationToken ct = default) => throw new NotImplementedException();
        public ReadOnlyMemory<float> DecodeStep(int tokenId, CancellationToken ct = default) => throw new NotImplementedException();
        public void TruncateCache(int length) { }
        public void TrimToLast(int keep) { }
        public void ResetCache() { }
        public KVCacheSnapshot ExportCache(int[] promptTokenIds) => throw new NotImplementedException();
        public void ImportCache(KVCacheSnapshot snapshot) { }
        public void Dispose() { }
    }

    private sealed class Factory(Func<InferenceEngineContext, IInferenceEngine?> create, string? reason = null) : IInferenceEngineFactory
    {
        private readonly Func<InferenceEngineContext, IInferenceEngine?> _create = create;
        private readonly string? _reason = reason;
        public IInferenceEngine? TryCreate(InferenceEngineContext context, out string? reason)
        {
            reason = _reason;
            return _create(context);
        }
    }

    private sealed class Plugin(string name, params object[] capabilities) : IAcceleratorPlugin
    {
        public string Name => name;
        public string Description => $"{name} test plugin";
        public IReadOnlyList<object> Capabilities => capabilities;
    }

    [Fact]
    public void NullOrBlank_OrCpu_ReturnsNull()
    {
        var plugins = new IAcceleratorPlugin[] { new Plugin("ilgpu", new Factory(_ => new FakeEngine())) };
        Assert.Null(InferenceEngineResolver.Resolve(null, plugins, Ctx));
        Assert.Null(InferenceEngineResolver.Resolve("", plugins, Ctx));
        Assert.Null(InferenceEngineResolver.Resolve("   ", plugins, Ctx));
        Assert.Null(InferenceEngineResolver.Resolve("CPU", plugins, Ctx));
        Assert.Null(InferenceEngineResolver.Resolve("cpu", plugins, Ctx));
    }

    [Fact]
    public void Resolve_FindsPluginByName_CaseInsensitive()
    {
        var engines = new IAcceleratorPlugin[]
        {
            new Plugin("ilgpu", new Factory(_ => new FakeEngine())),
        };

        using var created = InferenceEngineResolver.Resolve("ILGPU", engines, Ctx);
        Assert.IsType<FakeEngine>(created);
    }

    [Fact]
    public void LegacyCuda_ResolvesToTheCanonicalIlgpuPlugin()
    {
        // A stored pre-rename session preset names "cuda"; the plugin now ships as "ilgpu" and
        // must still be found (case-insensitively) so old saves don't break.
        var engines = new IAcceleratorPlugin[] { new Plugin("ilgpu", new Factory(_ => new FakeEngine())) };

        using var created = InferenceEngineResolver.Resolve("cuda", engines, Ctx);
        Assert.IsType<FakeEngine>(created);
    }

    [Fact]
    public void LegacyCuda_RouteIsCaseInsensitive()
    {
        var engines = new IAcceleratorPlugin[] { new Plugin("ilgpu", new Factory(_ => new FakeEngine())) };

        using var upper = InferenceEngineResolver.Resolve("CUDA", engines, Ctx);
        using var mixed = InferenceEngineResolver.Resolve("Cuda", engines, Ctx);
        Assert.IsType<FakeEngine>(upper);
        Assert.IsType<FakeEngine>(mixed);
    }

    [Fact]
    public void UnknownName_Throws_ListingAvailable()
    {
        var engines = new IAcceleratorPlugin[] { new Plugin("ilgpu", new Factory(_ => new FakeEngine())) };
        var ex = Assert.Throws<InvalidOperationException>(() => InferenceEngineResolver.Resolve("nope", engines, Ctx));
        Assert.Contains("nope", ex.Message);
        Assert.Contains("ilgpu", ex.Message);
    }

    [Fact]
    public void PluginWithoutInferenceFactory_Throws()
    {
        // IlgpuAcceleratorPlugin also carries a training factory; a plugin that offers none at all
        // (or only other capabilities) must be refused for inference explicitly.
        var engines = new IAcceleratorPlugin[] { new Plugin("ilgpu") };
        var ex = Assert.Throws<InvalidOperationException>(() => InferenceEngineResolver.Resolve("ilgpu", engines, Ctx));
        Assert.Contains("does not provide an inference engine", ex.Message);
    }

    [Fact]
    public void FactoryDeclining_ThrowsAcceleratorUnavailable_WithReason()
    {
        var engines = new IAcceleratorPlugin[] { new Plugin("ilgpu", new Factory(_ => null, "no CUDA device found")) };
        // The declined case is the consent-dialog signal: a subtype of InvalidOperationException
        // carrying the factory's reason, so the CUI knows to offer the picker rather than fail hard.
        var ex = Assert.Throws<AcceleratorUnavailableException>(() => InferenceEngineResolver.Resolve("ilgpu", engines, Ctx));
        Assert.Contains("no CUDA device found", ex.Message);
        Assert.Equal("no CUDA device found", ex.Reason);
    }

    [Fact]
    public void FactoryThatThrows_IsWrapped_NotLeaked()
    {
        var engines = new IAcceleratorPlugin[]
        {
            new Plugin("ilgpu", new Factory(_ => throw new InvalidOperationException("boom")))
        };
        var ex = Assert.Throws<InvalidOperationException>(() => InferenceEngineResolver.Resolve("ilgpu", engines, Ctx));
        // Unattributed plugin exceptions must not escape raw — the resolver wraps them with attribution.
        Assert.Contains("creating its inference engine", ex.Message);
    }
}
