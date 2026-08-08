using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
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
using SharpMind.Training.Loss;

namespace SharpMind.Tests.Training;

/// <summary>
/// Regression test: <see cref="Trainer"/> previously computed the loss gradient
/// but never invoked backprop, so optimizer steps ran on zero gradients and
/// training never moved the weights. The trainer must now drive the loss on a
/// held-out deterministic tweet downward.
/// </summary>
public sealed class TrainerRegressionTests : IDisposable
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
    public async Task TrainAsync_LossDescendsAcrossSteps()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        string path = _dir.Write("corpus.txt",
            string.Join('\n', Enumerable.Range(0, 40).Select(i => $"the quick fox jumps over the lazy dog number {i}")));

        var pipeline = CleaningPipeline
            .From(new TextFileSource(path))
            .Pipe(new NormaliseWhitespace());
        var loader = new DataLoader(pipeline, Tokenise,
            new PackingBatcher(batchSize: 2, maxSeqLen: 16));

        // Hold out one corpus-derived tweet for evaluation (still in-distribution).
        var evalIds = EvalTokenIds();
        var loss = new CrossEntropyLoss();
        float before = EvalLoss(model, loss, evalIds);

        var trainer = new Trainer(
            model,
            loader,
            new AdamW(model.Parameters(), lr: 0.05f, weightDecay: 0f),
            new ConstantScheduler(0.05f),
            loss,
            sharpConfig);

        await trainer.TrainAsync(totalSteps: 30);

        float after = EvalLoss(model, loss, evalIds);
        Assert.True(after < before, $"loss did not descend: {before:F4} → {after:F4}");
    }

    private static int[] Tokenise(string s) =>
        [.. s.Split(' ', StringSplitOptions.RemoveEmptyEntries)
             .Select(w => Math.Abs(w.GetHashCode()) % 16)];

    /// <summary>One corpus-shaped sequence, padded to [2, 8] for a valid batch.</summary>
    private static int[] EvalTokenIds()
    {
        var toks = Tokenise("the quick fox jumps over the lazy dog number 7 the quick fox over the lazy dog");
        return toks.Length >= 16 ? toks[..16] : toks.Concat(Enumerable.Repeat(toks[0], 16 - toks.Length)).ToArray();
    }

    private static float EvalLoss(Transformer model, ILoss<int> lossFn, int[] tokenIds)
    {
        var ids = tokenIds.AsSpan();
        using (var batchIds = Tensor<int>.From(ids.ToArray(), 2, ids.Length / 2))
        using (var flatIds = batchIds.Reshape(ids.Length))
        using (var logits = model.Forward(batchIds))
        using (var flatLogits = logits.Reshape(ids.Length, Cfg.VocabSize))
        {
            var shifted = new int[ids.Length];
            for (int i = 0; i + 1 < ids.Length; i++) shifted[i] = ids[i + 1];
            shifted[^1] = ids[0];
            using var flatLabels = Tensor<int>.From(shifted, ids.Length);
            return lossFn.Compute(flatLogits, flatLabels);
        }
    }
}