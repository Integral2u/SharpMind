using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Format.Conversion;
using SharpMind.Tokenization;
using SharpMind.Training;
using System.Text.Json.Nodes;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// Covers the parallel full-weight load (<c>maxParallelLoadDegree</c>).
///
/// The whole feature rests on one claim: fanning the load out across threads
/// must produce byte-identical weights to the sequential loop. The loaders
/// partition by the block a tensor resolves to so no two threads ever write the
/// same block, and keep the shared-field tensors (embedding, output head, final
/// norms) on the calling thread. These tests assert exactly that, for both
/// containers, on a model small enough to compare value-for-value.
/// </summary>
public class ParallelWeightLoadTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    // ── Degree resolution ───────────────────────────────────────────────────

    [Fact]
    public void DefaultDegree_IsAuto()
    {
        // A caller that never heard of this option gets the fan-out, not the old
        // sequential loop: the default is "one per core" (0), not 1.
        using var fixture = NewTinyModel();
        string path = Export(fixture, ".smm");

        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);
        var loader = new GgufLoader(qOps, path, fixture.Config);
        Assert.Equal(0, loader.MaxParallelLoadDegree);

        var smm = new SmmLoader(qOps, path, fixture.Config);
        Assert.Equal(0, smm.MaxParallelLoadDegree);

        // ...and 0 actually resolves to a fan-out on a multi-core host, rather
        // than silently degrading to sequential.
        int expected = Math.Min(Environment.ProcessorCount, 4096);
        if (Environment.ProcessorCount > 1)
            Assert.Equal(expected, ParallelTensorLoad.ResolveDegree(loader.MaxParallelLoadDegree, workItemCount: 4096, safeIo: false));
    }

    [Fact]
    public void ExplicitDegreeOne_StaysSequential()
    {
        // 1 is the escape hatch and has to keep working, both as an explicit
        // argument and as a property someone set deliberately.
        using var fixture = NewTinyModel();
        string path = Export(fixture, ".smm");

        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);
        var loader = new GgufLoader(qOps, path, fixture.Config, maxParallelLoadDegree: 1);
        Assert.Equal(1, loader.MaxParallelLoadDegree);
        Assert.Equal(0, ParallelTensorLoad.ResolveDegree(loader.MaxParallelLoadDegree, workItemCount: 4096, safeIo: false));

        var smm = new SmmLoader(qOps, path, fixture.Config, maxParallelLoadDegree: 1);
        Assert.Equal(1, smm.MaxParallelLoadDegree);
        Assert.Equal(0, ParallelTensorLoad.ResolveDegree(smm.MaxParallelLoadDegree, workItemCount: 4096, safeIo: false));
    }

    [Fact]
    public void ExplicitDegreeFour_FansOut()
    {
        // The other escape direction: an explicit N above 1 must still fan out,
        // so pinning a degree is not silently ignored.
        using var fixture = NewTinyModel();
        string path = Export(fixture, ".smm");

        var qOps = QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware);
        var loader = new GgufLoader(qOps, path, fixture.Config, maxParallelLoadDegree: 4);
        Assert.Equal(4, loader.MaxParallelLoadDegree);

        int expected = Math.Min(4, Math.Min(Environment.ProcessorCount, 4096));
        Assert.Equal(expected, ParallelTensorLoad.ResolveDegree(loader.MaxParallelLoadDegree, workItemCount: 4096, safeIo: false));
    }

    [Fact]
    public void ResolveDegree_OneOrFewerWorkItems_StaysSequential()
    {
        Assert.Equal(0, ParallelTensorLoad.ResolveDegree(requested: 4, workItemCount: 1, safeIo: false));
        Assert.Equal(0, ParallelTensorLoad.ResolveDegree(requested: 1, workItemCount: 64, safeIo: false));
        Assert.Equal(0, ParallelTensorLoad.ResolveDegree(requested: 0, workItemCount: 1, safeIo: false));
    }

    [Fact]
    public void ResolveDegree_ClampsToCoresAndWorkItems()
    {
        int cores = Environment.ProcessorCount;
        int expected = Math.Min(4, Math.Min(cores, 32));
        Assert.Equal(expected, ParallelTensorLoad.ResolveDegree(requested: 4, workItemCount: 32, safeIo: false));

        // More workers than blocks would only open idle file views.
        Assert.Equal(3, ParallelTensorLoad.ResolveDegree(requested: 16, workItemCount: 3, safeIo: false));
    }

    [Fact]
    public void ResolveDegree_ZeroMeansOnePerCore()
    {
        int expected = Math.Min(Environment.ProcessorCount, 32);
        Assert.Equal(expected, ParallelTensorLoad.ResolveDegree(requested: 0, workItemCount: 32, safeIo: false));
    }

    [Fact]
    public void ResolveDegree_SafeIoStaysSequential()
    {
        // Safe-IO is the WASM/browser path. It has no shared memory and no real
        // threads to overlap with, and ProcessorCount there reports the host's
        // cores — so a caller asking for 0 (auto) must still get the sequential
        // loop rather than N file views on a scheduler that cannot run them.
        Assert.Equal(0, ParallelTensorLoad.ResolveDegree(requested: 0, workItemCount: 4096, safeIo: true));
        Assert.Equal(0, ParallelTensorLoad.ResolveDegree(requested: 4, workItemCount: 4096, safeIo: true));
    }

    [Fact]
    public void ResolveDegree_Negative_IsRejected()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ParallelTensorLoad.ResolveDegree(requested: -1, workItemCount: 8, safeIo: false));
    }

    // ── Partitioning ────────────────────────────────────────────────────────

    [Fact]
    public void PartitionByWeight_KeepsEveryItemExactlyOnce()
    {
        var items = Enumerable.Range(0, 37).ToList();
        var buckets = ParallelTensorLoad.PartitionByWeight(items, 4, i => (long)i);

        Assert.Equal(4, buckets.Count);
        var seen = buckets.SelectMany(b => b).OrderBy(x => x).ToList();
        Assert.Equal(items, seen);
    }

    [Fact]
    public void PartitionByWeight_BalancesByWeightNotByCount()
    {
        // One item dominates. A count-based split (index % degree) would deal the
        // six small items out evenly and still leave the load as slow as the
        // single heavy bucket; greedy-by-weight parks the 1000 alone.
        var items = new List<int> { 1000, 1, 1, 1, 1, 1, 1 };
        var buckets = ParallelTensorLoad.PartitionByWeight(items, 4, i => (long)i);

        var loads = buckets.Select(b => b.Sum(i => (long)i)).OrderBy(x => x).ToList();
        Assert.Equal([2L, 2L, 2L, 1000L], loads);

        // The heavy item is never co-bucketed with small work.
        var heavy = buckets.Single(b => b.Contains(1000));
        Assert.Single(heavy);
    }

    [Fact]
    public void PartitionByWeight_SpreadsEqualWeightsOnePerBucket()
    {
        var buckets = ParallelTensorLoad.PartitionByWeight([1, 2, 3, 4], 4, i => (long)i);
        Assert.All(buckets, b => Assert.Single(b));
    }

    [Fact]
    public void PartitionByWeight_DegreeExceedingItems_LeavesEmptyBuckets()
    {
        var buckets = ParallelTensorLoad.PartitionByWeight([1, 2], 4, i => (long)i);
        Assert.Equal(4, buckets.Count);
        Assert.Equal(2, buckets.Sum(b => b.Count));
    }

    // ── End-to-end parity ───────────────────────────────────────────────────

    [Theory]
    [InlineData(".smm")]
    [InlineData(".gguf")]
    public void ParallelLoad_ProducesIdenticalWeightsToSequential(string format)
    {
        using var fixture = NewTinyModel();
        string path = Export(fixture, format);

        using var sequential = Load(path, fixture.Config, fixture.SharpConfig, degree: 1);
        using var parallel = Load(path, fixture.Config, fixture.SharpConfig, degree: 0); // 0 = one per core

        AssertWeightsIdentical(sequential, parallel);
    }

    [Theory]
    [InlineData(".smm")]
    [InlineData(".gguf")]
    public void ParallelLoad_FillsProgressMonotonicallyToOne(string format)
    {
        using var fixture = NewTinyModel();
        string path = Export(fixture, format);

        // The collector itself must be thread-safe: the callback now runs under a
        // lock inside the loader, but a bare List.Add here would be the race the
        // test is meant to catch rather than the one it catches.
        var seen = new List<float>();
        var gate = new Lock();
        using var weights = Load(path, fixture.Config, fixture.SharpConfig, degree: 0, progress: v =>
        {
            lock (gate) seen.Add(v);
        });

        Assert.NotEmpty(seen);
        Assert.Equal(1f, seen[^1]);
        for (int i = 1; i < seen.Count; i++)
            Assert.True(seen[i] >= seen[i - 1], $"progress went backwards: {seen[i - 1]} -> {seen[i]}");
    }

    [Theory]
    [InlineData(".smm")]
    [InlineData(".gguf")]
    public async Task ParallelLoad_HonoursCancellation(string format)
    {
        using var fixture = NewTinyModel();
        string path = Export(fixture, format);

        using var cts = new CancellationTokenSource();
        cts.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            using var weights = ModelFactory.CreateWeights(
                fixture.Config, fixture.SharpConfig, QuantizationFactory.Create(fixture.SharpConfig.ResolvedHardware),
                path, LoadMode.Full, maxParallelLoadDegree: 0);
            await Task.Run(() => weights.InitializeWeights(cancellationToken: cts.Token), cts.Token);
        });
    }

    [Theory]
    [InlineData(".smm")]
    [InlineData(".gguf")]
    public void ParallelLoad_PopulatesTheSameTensorMetadataAsSequential(string format)
    {
        using var fixture = NewTinyModel();
        string path = Export(fixture, format);

        using var sequential = Load(path, fixture.Config, fixture.SharpConfig, degree: 1);
        using var parallel = Load(path, fixture.Config, fixture.SharpConfig, degree: 0);

        Assert.Equal(sequential.Blocks.Length, parallel.Blocks.Length);
        for (int i = 0; i < sequential.Blocks.Length; i++)
        {
            var a = sequential.Blocks[i].TensorMeta;
            var b = parallel.Blocks[i].TensorMeta;
            Assert.Equal(a.Count, b.Count);
            foreach (var (key, meta) in a)
            {
                Assert.True(b.TryGetValue(key, out var other), $"blk{i} lost tensor meta '{key}'");
                Assert.Equal(meta, other);
            }
        }

        Assert.Equal(sequential.RawEmbeddingDtype, parallel.RawEmbeddingDtype);
        Assert.Equal(sequential.RawLmHeadDtype, parallel.RawLmHeadDtype);
        Assert.Equal(sequential.HasLmHead, parallel.HasLmHead);
    }

    [Fact]
    public void ParallelLoad_MoE_BindsEveryExpertExactlyOnce()
    {
        // The MoE per-expert raw fields and dtypes live in plain dictionaries on
        // the block, initialised with ??= — exactly the state two threads racing
        // on one block would corrupt. Partitioning by block must prevent that.
        using var fixture = NewMoEModel();
        string smmPath = Path.Combine(_temp.Path, "moe.smm");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, smmPath,
            new SmmWriteOptions { Source = "training" }, model: fixture.Model);
        string ggufPath = Path.Combine(_temp.Path, "moe.gguf");
        SmmToGufConverter.Convert(smmPath, ggufPath);

        // Sanity: the sequential load must agree, otherwise a failure below is
        // about the export rather than about the fan-out.
        foreach (string path in new[] { smmPath, ggufPath })
        {
            foreach (int degree in new[] { 1, 0 })
            {
                using var weights = Load(path, fixture.Config, fixture.SharpConfig, degree);

                for (int i = 0; i < weights.Blocks.Length; i++)
                {
                    var block = weights.Blocks[i];
                    string what = $"{Path.GetFileName(path)} degree={degree} blk{i}";
                    Assert.Equal(fixture.Config.NumExperts, block.RawWgateExp?.Count ?? 0);
                    Assert.Equal(fixture.Config.NumExperts, block.RawWupExp?.Count ?? 0);
                    Assert.Equal(fixture.Config.NumExperts, block.RawWdownExp?.Count ?? 0);
                    Assert.Equal(fixture.Config.NumExperts, block.QuantDtypeWgateExp?.Count ?? 0);
                    Assert.Equal(fixture.Config.NumExperts, block.QuantDtypeWdownExp?.Count ?? 0);
                    for (int e = 0; e < fixture.Config.NumExperts; e++)
                        Assert.True(block.RawWdownExp![e].Length > 0, $"{what} expert {e} has no bytes");
                }
            }
        }
    }

    [Fact]
    public void ParallelLoad_RepeatedRuns_AreStable()
    {
        // Fan-out order is nondeterministic; the resulting weights must not be.
        using var fixture = NewTinyModel();
        string path = Export(fixture, ".smm");

        using var first = Load(path, fixture.Config, fixture.SharpConfig, degree: 0);
        using var second = Load(path, fixture.Config, fixture.SharpConfig, degree: 0);

        AssertWeightsIdentical(first, second);
        AssertWeightsIdentical(first, Load(path, fixture.Config, fixture.SharpConfig, degree: 1));
    }

    // ── Progress ordering ────────────────────────────────────────────────────

    [Fact]
    public void ProgressReporter_DropsOutOfOrderReports()
    {
        // Workers increment a shared counter, so a thread that took 6 can reach
        // the callback before one that took 5. The sequential loop never showed a
        // caller a rewind, and this is the only thing standing between a fan-out
        // and a progress bar that jumps backwards — so pin it directly, where it
        // is deterministic, rather than hoping the load test interleaves that way.
        var seen = new List<float>();
        var reporter = new ParallelTensorLoad.ProgressReporter(new InlineProgress(seen.Add), total: 10);

        reporter.ReportLoaded(5);
        reporter.ReportLoaded(6); // arrives second, but is further along
        reporter.ReportLoaded(6); // duplicate from a retried counter read
        reporter.ReportLoaded(4); // genuinely late: must be dropped
        reporter.ReportLoaded(10);

        Assert.Equal([0.5f, 0.6f, 1f], seen);

        // A 100% already reported must not be reported twice.
        reporter.ReportComplete();
        Assert.Equal([0.5f, 0.6f, 1f], seen);
    }

    [Fact]
    public void ProgressReporter_NullInnerIsANoOp()
    {
        var reporter = new ParallelTensorLoad.ProgressReporter(null, total: 8);
        reporter.ReportLoaded(1);
        reporter.ReportLoaded(8);
        reporter.ReportComplete();
    }

    [Fact]
    public void ProgressReporter_ZeroTotalCompletesRatherThanDividingByZero()
    {
        var seen = new List<float>();
        var reporter = new ParallelTensorLoad.ProgressReporter(new InlineProgress(seen.Add), total: 0);
        reporter.ReportLoaded(0);
        Assert.Equal([1f], seen);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────
    private string Export(TinyModelFixture fixture, string format)
    {
        string path = Path.Combine(_temp.Path, $"parallel{format}");
        SmmTrainingExporter.Export(fixture.Weights, fixture.Tokenizer, path, new SmmWriteOptions { Source = "training" });
        if (format == ".gguf")
        {
            string gguf = Path.Combine(_temp.Path, "parallel-converted.gguf");
            SmmToGufConverter.Convert(path, gguf);
            return gguf;
        }
        return path;
    }

    private static TransformerWeights Load(
        string path, ModelConfig config, SharpMindConfig sharpConfig, int degree, Action<float>? progress = null)
    {
        var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
        var weights = ModelFactory.CreateWeights(config, sharpConfig, qOps, path, LoadMode.Full,
            maxParallelLoadDegree: degree);
        weights.InitializeWeights(progress is null ? null : new InlineProgress(progress));
        return weights;
    }

    private static void AssertWeightsIdentical(TransformerWeights a, TransformerWeights b)
    {
        // An F32 export makes this an exact comparison, not a tolerance one:
        // any tensor the fan-out dropped, doubled or interleaved shows up here.
        AssertTensorIdentical(a.EmbeddingWeight, b.EmbeddingWeight, "token_embd");
        AssertTensorIdentical(a.FinalNormWeight, b.FinalNormWeight, "output_norm");
        Assert.Equal(a.RawEmbedding, b.RawEmbedding);
        Assert.Equal(a.RawLmHead, b.RawLmHead);
        Assert.Equal(a.Blocks.Length, b.Blocks.Length);

        for (int i = 0; i < a.Blocks.Length; i++)
        {
            var x = a.Blocks[i];
            var y = b.Blocks[i];
            string blk = $"blk{i}";
            AssertTensorIdentical(x.Wq, y.Wq, $"{blk}.attn_q");
            AssertTensorIdentical(x.Wk, y.Wk, $"{blk}.attn_k");
            AssertTensorIdentical(x.Wv, y.Wv, $"{blk}.attn_v");
            AssertTensorIdentical(x.Wo, y.Wo, $"{blk}.attn_output");
            AssertTensorIdentical(x.Wf1, y.Wf1, $"{blk}.ffn_up");
            AssertTensorIdentical(x.Wf2, y.Wf2, $"{blk}.ffn_down");
            AssertTensorIdentical(x.Norm1W, y.Norm1W, $"{blk}.attn_norm");
            AssertTensorIdentical(x.Norm2W, y.Norm2W, $"{blk}.ffn_norm");
            Assert.Equal(x.RawWq, y.RawWq);
            Assert.Equal(x.RawWk, y.RawWk);
            Assert.Equal(x.RawWv, y.RawWv);
            Assert.Equal(x.RawWo, y.RawWo);
            Assert.Equal(x.RawWgate, y.RawWgate);
            Assert.Equal(x.RawWup, y.RawWup);
            Assert.Equal(x.RawWf1, y.RawWf1);
            Assert.Equal(x.RawWf2, y.RawWf2);
        }
    }

    private static void AssertTensorIdentical(Tensor<float>? a, Tensor<float>? b, string label)
    {
        if (a is null || b is null)
        {
            Assert.True(a is null && b is null, $"{label}: one side is null ({a is null} vs {b is null})");
            return;
        }
        Assert.Equal(a.Shape, b.Shape);
        Assert.Equal(a.Data, b.Data);
    }

    private sealed class InlineProgress(Action<float> onReport) : IProgress<float>
    {
        public void Report(float value) => onReport(value);
    }

    private sealed class TinyModelFixture : IDisposable
    {
        public required TransformerWeights Weights { get; init; }
        public required Transformer Model { get; init; }
        public required ModelConfig Config { get; init; }
        public required SharpMindConfig SharpConfig { get; init; }
        public required Tokenizer Tokenizer { get; init; }
        public void Dispose()
        {
            Model.Dispose();
            Weights.Dispose();
        }
    }

    /// <summary>
    /// A tiny F32 model. F32 keeps the parity assertions exact, so the tests
    /// measure "did the fan-out lose or duplicate a tensor" rather than
    /// re-measuring quantization error.
    /// </summary>
    private static TinyModelFixture NewTinyModel() => BuildFixture(ModelConfig.Learnable, 20260805);

    private static TinyModelFixture NewMoEModel()
        => BuildFixture(ModelConfig.Learnable with { NumExperts = 4, TopKExperts = 2 }, 777);

    private static TinyModelFixture BuildFixture(ModelConfig config, int seed)
    {
        var sharpConfig = config.NumExperts > 0
            ? SharpMindConfig.ForModel(config.NumHeads, config.NumKvHeads, "mixtral") with { Hardware = HardwareTier.Scalar }
            : SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(config, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, seed);
        var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        if (config.NumExperts > 0)
            WeightInitializer.InitializeModelMoE(model, seed + 1013);
        return new TinyModelFixture
        {
            Weights = weights,
            Model = model,
            Config = config,
            SharpConfig = sharpConfig,
            Tokenizer = BuildTokenizer(),
        };
    }

    private static Tokenizer BuildTokenizer()
    {
        var vocabObj = new JsonObject();
        int vocab = ModelConfig.Learnable.VocabSize;
        for (int i = 0; i < vocab - 4; i++) vocabObj[$"w{i}"] = i;
        vocabObj["<unk>"] = vocab - 4;
        vocabObj["<s>"] = vocab - 3;
        vocabObj["</s>"] = vocab - 2;
        vocabObj["<pad>"] = vocab - 1;

        var root = new JsonObject
        {
            ["version"] = "1.0",
            ["pre_tokenizer"] = "whitespace",
            ["special_tokens"] = new JsonObject
            {
                ["unk"] = "<unk>",
                ["bos"] = "<s>",
                ["eos"] = "</s>",
                ["pad"] = "<pad>",
                ["additional"] = new JsonArray(),
            },
            ["vocab"] = vocabObj,
            ["merges"] = new JsonArray(),
        };
        return Tokenizer.FromJson(root.ToJsonString());
    }
}
