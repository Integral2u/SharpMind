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

/// <summary>
/// The data pipeline streams a corpus exactly once; a small corpus therefore
/// yields only a handful of batches. With the DataLoader's batch budget the
/// TrainLoop must re-enumerate the pipeline (epoch-style) until TotalSteps is
/// genuinely reached instead of stopping early at the last available batch.
/// </summary>
public sealed class TrainLoopEpochTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private static ModelConfig Cfg => new()
    {
        VocabSize = 16,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 16,
    };

    [Fact]
    public async Task RunAsync_TotalStepsBeyondSinglePass_ReachesRequestedStep()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        // Tiny corpus: a single pass yields far fewer batches than TotalSteps,
        // so without the batch budget the loop would stop early.
        string path = _dir.Write("tiny.txt",
            string.Join('\n', Enumerable.Range(0, 4).Select(i => $"the quick fox jumps over the lazy dog {i}")));
        var pipeline = CleaningPipeline.From(new TextFileSource(path)).Pipe(new NormaliseWhitespace());
        var loader = new DataLoader(pipeline, s => Tokenise(s),
            new PackingBatcher(batchSize: 2, maxSeqLen: 16),
            maxBatches: 20);

        int lastStep = -1;
        var loop = new TrainLoop(
            model,
            model.Parameters(),
            loader,
            new AdamW(model.Parameters(), lr: 0.01f, weightDecay: 0f),
            new ConstantScheduler(0.01f),
            TrainingOpsFactory.Create(sharpConfig),
            smmConfig: sharpConfig,
            config: new TrainConfig { TotalSteps = 20, GradAccumSteps = 1, LogInterval = 1 });

        await loop.RunAsync(onStep: r => lastStep = r.Step);

        Assert.Equal(20, lastStep);
    }

    private static int[] Tokenise(string s) => TestTokens.Encode(s, 16);
}
