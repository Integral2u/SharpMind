using SharpMind.Core;
using SharpMind.Core.Training;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Loss;
using SharpMind.Training.Optimizers;

namespace SharpMind.Tests.Training;

/// <summary>
/// Covers the finite-difference trainer: loss must descend, weights must move
/// (the regression for the fresh-Grad-per-call bug), progress must be
/// monotonic, and the step budget must be honoured.
/// </summary>
public class FiniteDifferenceTrainerTests
{
    private static readonly ModelConfig Config = new()
    {
        VocabSize = 64,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 1,
        NumKvHeads = 1,
        FfnDim = 16,
        MaxSeqLen = 512,
    };

    private static (Transformer Model, LearnableGenerator Generator) Fixture(int seed = 1234)
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Config, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, seed);
        var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        var generator = new LearnableGenerator(new LearnableConfig(), new Random(seed));
        return (model, generator);
    }

    private static (FiniteDifferenceTrainer Trainer, IReadOnlyList<Parameter> Parameters) CreateTrainer(
        Transformer model, LearnableGenerator generator, int totalSteps, int logInterval)
    {
        var parameters = model.Parameters().ToList();
        var optimizer = new AdamW(parameters, lr: 0.02f, weightDecay: 0f);
        var trainer = new FiniteDifferenceTrainer(
            model,
            generator.ToTrainingBatches(batchSize: 4, seqLen: 3),
            optimizer,
            parameters: parameters,
            loss: new CrossEntropyLoss(),
            config: new FiniteDifferenceConfig { TotalSteps = totalSteps, LogInterval = logInterval });
        return (trainer, parameters);
    }

    [Fact]
    public async Task TrainAsync_ReducesLossAndMovesWeights()
    {
        var (model, generator) = Fixture();
        using var _ = model;
        var (trainer, parameters) = CreateTrainer(model, generator, totalSteps: 12, logInterval: 1);
        float before = parameters[0].Data.Data[0];

        var losses = new List<float>();
        var result = await trainer.TrainAsync(onStep: step => losses.Add(step.Loss));

        Assert.Equal(12, result.Steps);
        Assert.NotEmpty(losses);
        Assert.True(result.FinalLoss < losses[0], $"loss did not descend: {losses[0]:F4} → {result.FinalLoss:F4}");
        Assert.True(float.IsFinite(result.FinalLoss));
        Assert.NotEqual(before, parameters[0].Data.Data[0]);
    }

    [Fact]
    public async Task TrainAsync_ReportsMonotonicProgressToFull()
    {
        var (model, generator) = Fixture();
        using var _ = model;
        var (trainer, _) = CreateTrainer(model, generator, totalSteps: 8, logInterval: 8);
        var progress = new ListProgress();

        var result = await trainer.TrainAsync(progress: progress);

        Assert.Equal(8, result.Steps);
        Assert.Equal(8, progress.Values.Count);
        Assert.True(progress.Values[0] > 0f);
        Assert.Equal(1f, progress.Values[^1], 3);
        for (int i = 1; i < progress.Values.Count; i++)
            Assert.True(progress.Values[i] >= progress.Values[i - 1], "progress must be monotonic");
    }

    private sealed class ListProgress : IProgress<float>
    {
        public List<float> Values { get; } = new();
        public void Report(float value) => Values.Add(value);
    }
}
