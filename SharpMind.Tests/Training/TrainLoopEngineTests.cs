using SharpMind.Core;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;

namespace SharpMind.Tests.Training;

/// <summary>An engine that does no compute: proves TrainLoop drives whatever engine it is given.</summary>
internal sealed class CountingEngine : ITrainingEngine
{
    public int Calls { get; private set; }
    public bool Disposed { get; private set; }
    public float ForwardBackward(TrainingBatch batch, CancellationToken cancellationToken = default)
    {
        Calls++;
        return 1.5f;
    }
    public void Dispose() => Disposed = true;
}

public sealed class TrainLoopEngineTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private static ModelConfig Cfg => new()
    {
        VocabSize = 16, HiddenDim = 8, NumLayers = 1, NumHeads = 2, NumKvHeads = 2, FfnDim = 16, MaxSeqLen = 16,
    };

    private DataLoader Loader()
    {
        string path = _dir.Write("corpus.txt",
            string.Join('\n', Enumerable.Range(0, 40).Select(i => $"the quick fox jumps over the lazy dog number {i}")));
        var pipeline = CleaningPipeline.From(new TextFileSource(path)).Pipe(new NormaliseWhitespace());
        return new DataLoader(pipeline, s => TestTokens.Encode(s, 16), new PackingBatcher(batchSize: 2, maxSeqLen: 16));
    }

    [Fact]
    public async Task RunAsync_UsesTheInjectedEngine_OncePerBatch_AndDoesNotDisposeIt()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        var engine = new CountingEngine();
        var losses = new List<float>();

        var loop = new TrainLoop(
            model,
            model.Parameters(),
            Loader(),
            new AdamW(model.Parameters(), lr: 0.01f, weightDecay: 0f),
            new ConstantScheduler(0.01f),
            TrainingOpsFactory.Create(sharpConfig),
            smmConfig: sharpConfig,
            config: new TrainConfig { TotalSteps = 3, GradAccumSteps = 2, LogInterval = 1, CheckpointInterval = 0 },
            engine: engine);

        await loop.RunAsync(onStep: r => losses.Add(r.Loss));

        Assert.Equal(6, engine.Calls);                         // 3 steps × 2 accumulated batches
        Assert.Equal(new[] { 1.5f, 1.5f, 1.5f }, losses);      // mean of the engine's constant loss
        Assert.False(engine.Disposed);                          // caller owns the engine
    }
}
