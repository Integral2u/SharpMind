using SharpMind.Core;
using SharpMind.Core.Plugins;
using SharpMind.Core.Training;
using SharpMind.Data.Batching;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Loss;

namespace SharpMind.Tests.Training;

public sealed class TrainingEngineResolverTests : IDisposable
{
    private sealed class Plugin(string name, params object[] capabilities) : IAcceleratorPlugin
    {
        public string Name => name;
        public string Description => "test";
        public IReadOnlyList<object> Capabilities => capabilities;
    }

    private sealed class Factory(Func<TrainingEngineContext, (ITrainingEngine?, string?)> create) : ITrainingEngineFactory
    {
        public ITrainingEngine? TryCreate(TrainingEngineContext context, out string? reason)
        {
            var (engine, why) = create(context);
            reason = why;
            return engine;
        }
    }

    private sealed class NullEngine : ITrainingEngine
    {
        public float ForwardBackward(TrainingBatch batch, CancellationToken cancellationToken = default) => 0f;
        public void Dispose() { }
    }

    private readonly Transformer _model;
    private readonly TrainingEngineContext _ctx;

    public TrainingEngineResolverTests()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var cfg = new ModelConfig { VocabSize = 16, HiddenDim = 8, NumLayers = 1, NumHeads = 2, NumKvHeads = 2, FfnDim = 16, MaxSeqLen = 16 };
        var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        _model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        _ctx = new TrainingEngineContext(_model, _model.Parameters().ToList(), GradientMappingFactory.Create(sharpConfig),
            sharpConfig, new CrossEntropyLoss(), BatchSize: 2, SeqLen: 16);
    }

    public void Dispose() => _model.Dispose();

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    [InlineData("CPU")]
    [InlineData("cpu")]
    public void NullBlankOrCpu_ReturnsNull_MeaningDefaultEngine(string? accelerator)
    {
        var plugins = new List<IAcceleratorPlugin> { new Plugin("cuda", new Factory(_ => (new NullEngine(), null))) };

        Assert.Null(TrainingEngineResolver.Resolve(accelerator, plugins, _ctx));
    }

    [Fact]
    public void KnownName_ReturnsTheFactoryEngine_CaseInsensitive()
    {
        var engine = new NullEngine();
        var plugins = new List<IAcceleratorPlugin> { new Plugin("cuda", new Factory(_ => (engine, null))) };

        Assert.Same(engine, TrainingEngineResolver.Resolve("CUDA", plugins, _ctx));
    }

    [Fact]
    public void UnknownName_Throws_ListingWhatIsAvailable()
    {
        var plugins = new List<IAcceleratorPlugin> { new Plugin("cuda"), new Plugin("ilgpu") };

        var ex = Assert.Throws<InvalidOperationException>(() => TrainingEngineResolver.Resolve("metal", plugins, _ctx));

        Assert.Contains("'metal'", ex.Message);
        Assert.Contains("cuda", ex.Message);
        Assert.Contains("ilgpu", ex.Message);
    }

    [Fact]
    public void PluginWithoutTrainingCapability_Throws()
    {
        var plugins = new List<IAcceleratorPlugin> { new Plugin("cuda", "not a factory") };

        var ex = Assert.Throws<InvalidOperationException>(() => TrainingEngineResolver.Resolve("cuda", plugins, _ctx));

        Assert.Contains("does not provide a training engine", ex.Message);
    }

    [Fact]
    public void FactoryDeclines_Throws_WithItsReason()
    {
        var plugins = new List<IAcceleratorPlugin> { new Plugin("cuda", new Factory(_ => (null, "MoE layers are not supported"))) };

        var ex = Assert.Throws<InvalidOperationException>(() => TrainingEngineResolver.Resolve("cuda", plugins, _ctx));

        Assert.Contains("MoE layers are not supported", ex.Message);
    }

    [Fact]
    public void FactoryThrows_Throws_AttributedToThePlugin_WithTheOriginalAsInnerException()
    {
        var thrown = new InvalidOperationException("driver not found");
        var plugins = new List<IAcceleratorPlugin> { new Plugin("cuda", new Factory(_ => throw thrown)) };

        var ex = Assert.Throws<InvalidOperationException>(() => TrainingEngineResolver.Resolve("cuda", plugins, _ctx));

        Assert.Contains("cuda", ex.Message);
        Assert.Contains("driver not found", ex.Message);
        Assert.Same(thrown, ex.InnerException);
    }
}
