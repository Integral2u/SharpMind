using SharpMind.Core;
using SharpMind.Core.Tensors;
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
/// Verifies <see cref="TrainConfig.KeepRecent"/> caps the number of retained
/// rolling step-* checkpoint directories during a run: newest N survive, the
/// final "-final" checkpoint is never pruned, and a negative KeepRecent
/// disables rolling checkpoints entirely.
/// </summary>
public sealed class CheckpointRetentionTests : IDisposable
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

    private DataLoader Loader(int docs)
    {
        string path = _dir.Write("corpus.txt",
            string.Join('\n', Enumerable.Range(0, docs).Select(i => $"the quick fox jumps over the lazy dog number {i}")));
        var pipeline = CleaningPipeline.From(new TextFileSource(path)).Pipe(new NormaliseWhitespace());
        return new DataLoader(pipeline, s => Tokenise(s), new PackingBatcher(batchSize: 2, maxSeqLen: 16));
    }

    private static int[] Tokenise(string s) =>
        [.. s.Split(' ', System.StringSplitOptions.RemoveEmptyEntries)
             .Select(w => Math.Abs(w.GetHashCode()) % 16)];

    [Fact]
    public async Task KeepRecent2_LeavesTwoRollingPlusFinal()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        string ckDir = _dir.Path;
        var loop = new TrainLoop(
            model,
            model.Parameters(),
            Loader(docs: 40),
            new AdamW(model.Parameters(), lr: 0.01f, weightDecay: 0f),
            new ConstantScheduler(0.01f),
            TrainingOpsFactory.Create(sharpConfig),
            smmConfig: sharpConfig,
            config: new TrainConfig
            {
                TotalSteps = 10,
                GradAccumSteps = 1,
                CheckpointInterval = 2,
                CheckpointDir = ckDir,
                KeepRecent = 2,
            });

        await loop.RunAsync();

        var rolling = Directory.GetDirectories(ckDir, "step-*")
            .Where(d => !d.EndsWith("-final", System.StringComparison.Ordinal))
            .ToList();
        var finals = Directory.GetDirectories(ckDir, "*-final")
            .ToList();
        Assert.Equal(2, rolling.Count);             // newest two survive
        Assert.Single(finals);                       // final always kept
        Assert.Contains(Path.Combine(ckDir, "step-0000010-final"), finals);
    }

    [Fact]
    public async Task KeepRecentNegative_KeepsOnlyFinal()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        string ckDir = _dir.Path;
        var loop = new TrainLoop(
            model,
            model.Parameters(),
            Loader(docs: 40),
            new AdamW(model.Parameters(), lr: 0.01f, weightDecay: 0f),
            new ConstantScheduler(0.01f),
            TrainingOpsFactory.Create(sharpConfig),
            smmConfig: sharpConfig,
            config: new TrainConfig
            {
                TotalSteps = 10,
                GradAccumSteps = 1,
                CheckpointInterval = 2,
                CheckpointDir = ckDir,
                KeepRecent = -1,
            });

        await loop.RunAsync();

        var all = Directory.GetDirectories(ckDir, "step-*").ToList();
        Assert.Single(all);            // only the final checkpoint
        Assert.EndsWith("-final", Path.GetFileName(all[0]));
    }
}