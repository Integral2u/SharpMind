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
/// A strict resume onto a model built for a different vocabulary cannot work:
/// the weight-tied embedding is shape-bound to the token count. The loop must
/// fail with an actionable message pointing at the warm-restart path instead of
/// leaking a bare checkpoint shape-mismatch.
/// </summary>
public sealed class GuardVocabTests : IDisposable
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
    public async Task Resume_WithDifferentRecordedVocab_ThrowsActionableError()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        // A checkpoint whose weights match this model, but whose metadata claims
        // it was trained on a different vocabulary size (e.g. before a grow).
        string checkpoint = Path.Combine(_dir.Path, "ckpt");
        Checkpoint.Save(checkpoint, model.Parameters(), step: 3,
            extra: new Dictionary<string, string> { ["vocabSize"] = "24" });

        string path = _dir.Write("tiny.txt", "the quick fox jumps over the lazy dog");
        var pipeline = CleaningPipeline.From(new TextFileSource(path)).Pipe(new NormaliseWhitespace());
        var loader = new DataLoader(pipeline, s => TestTokens.Encode(s, 16),
            new PackingBatcher(batchSize: 2, maxSeqLen: 16), maxBatches: 4);

        var loop = new TrainLoop(
            model,
            model.Parameters(),
            loader,
            new AdamW(model.Parameters(), lr: 0.01f, weightDecay: 0f),
            new ConstantScheduler(0.01f),
            TrainingOpsFactory.Create(sharpConfig),
            smmConfig: sharpConfig,
            config: new TrainConfig
            {
                TotalSteps = 1,
                GradAccumSteps = 1,
                ResumeFrom = checkpoint,
                ExpectedVocabSize = 16,
            });

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => loop.RunAsync());
        Assert.Contains("vocabulary size", ex.Message);
        Assert.Contains("warm restart", ex.Message);
    }

    [Fact]
    public async Task Resume_WithMatchingVocab_DoesNotThrow()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        string checkpoint = Path.Combine(_dir.Path, "ckpt-ok");
        Checkpoint.Save(checkpoint, model.Parameters(), step: 0,
            extra: new Dictionary<string, string> { ["vocabSize"] = "16" });

        string path = _dir.Write("tiny.txt", "the quick fox jumps over the lazy dog");
        var pipeline = CleaningPipeline.From(new TextFileSource(path)).Pipe(new NormaliseWhitespace());
        var loader = new DataLoader(pipeline, s => TestTokens.Encode(s, 16),
            new PackingBatcher(batchSize: 2, maxSeqLen: 16), maxBatches: 4);

        var loop = new TrainLoop(
            model,
            model.Parameters(),
            loader,
            new AdamW(model.Parameters(), lr: 0.01f, weightDecay: 0f),
            new ConstantScheduler(0.01f),
            TrainingOpsFactory.Create(sharpConfig),
            smmConfig: sharpConfig,
            config: new TrainConfig
            {
                TotalSteps = 1,
                GradAccumSteps = 1,
                ResumeFrom = checkpoint,
                ExpectedVocabSize = 16,
            });

        await loop.RunAsync();
    }
}
