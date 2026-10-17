using SharpMind.Core;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Loss;

namespace SharpMind.Tests.Data;

/// <summary>
/// B3: a validation set is built once and cached in memory, so evaluation is
/// fast (no re-tokenisation) and comparable across steps â€” two evaluations over
/// the cache produce the identical loss for the same weights.
/// </summary>
public sealed class CachedBatchSetTests : IDisposable
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

    private string CorpusPath()
        => _dir.Write("corpus.txt",
            string.Join('\n', Enumerable.Range(0, 40).Select(i => $"the quick fox jumps over the lazy dog {i}")));

    private DataLoader Loader(string path, int maxBatches)
    {
        var pipeline = CleaningPipeline.From(new TextFileSource(path)).Pipe(new NormaliseWhitespace());
        return new DataLoader(pipeline, s => TestTokens.Encode(s, 16),
            new PackingBatcher(batchSize: 2, maxSeqLen: 16),
            maxBatches: maxBatches);
    }

    private static string Fingerprint(TrainingBatch batch)
        => string.Join(",", batch.TokenIds.Data.ToArray());

    private Evaluator NewEvaluator(out Transformer model)
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 1234);
        model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        var loss = new CrossEntropyLoss(ignoreId: -100, labelSmoothing: 0f);
        return new Evaluator(model, loss);
    }

    [Fact]
    public async Task LoadCachedAsync_MaterialisesTheRequestedNumberOfBatches()
    {
        using var cache = await Loader(CorpusPath(), maxBatches: 20).LoadCachedAsync(5);

        Assert.Equal(5, cache.Count);
        Assert.True(cache.TotalRealTokens > 0);
    }

    [Fact]
    public async Task TwoEvalsOfTheSameWeights_GiveIdenticalLoss()
    {
        using var cache = await Loader(CorpusPath(), maxBatches: 20).LoadCachedAsync(4);
        var evaluator = NewEvaluator(out var model);
        using var modelOwner = model;

        var first = evaluator.Evaluate(cache.Batches);
        var second = evaluator.Evaluate(cache.Batches);

        Assert.Equal(first.Loss, second.Loss);
        Assert.Equal(first.Perplexity, second.Perplexity);
    }

    [Fact]
    public async Task CachedBatches_MatchAFreshStream()
    {
        string path = CorpusPath();
        using var cache = await Loader(path, maxBatches: 20).LoadCachedAsync(5);

        var fresh = new List<string>();
        await foreach (var batch in Loader(path, maxBatches: 5).LoadAsync())
        {
            fresh.Add(Fingerprint(batch));
            batch.Dispose();
        }

        Assert.Equal(fresh, cache.Batches.Select(Fingerprint));
    }

    [Fact]
    public async Task CachedData_IsImmuneToOneTheFlySourceChanges()
    {
        string path = CorpusPath();
        using var cache = await Loader(path, maxBatches: 20).LoadCachedAsync(4);
        var before = cache.Batches.Select(Fingerprint).ToArray();

        // Rewrite the file entirely; the in-memory cache must not notice.
        File.WriteAllText(path, "completely different corpus\nwith unrelated documents\n");

        var evaluator = NewEvaluator(out var model);
        using var modelOwner = model;
        var afterEval = evaluator.Evaluate(cache.Batches);
        var firstEval = evaluator.Evaluate(cache.Batches);

        Assert.Equal(before, cache.Batches.Select(Fingerprint).ToArray());
        Assert.Equal(firstEval.Loss, afterEval.Loss);
    }

    [Fact]
    public async Task LoadCachedAsync_RejectsNonPositiveCount()
    {
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => Loader(CorpusPath(), maxBatches: 20).LoadCachedAsync(0));
    }
}
