using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// Batching is the next step for throughput (one sequence per model step caps a
/// server at one user's decode rate), and the risk is silent: a batched pass that
/// mixes sequences or indexes the KV cache by the wrong axis still produces
/// plausible logits. These pin the property that matters — a batch of N rows must
/// give exactly the answers N single-row passes give, for both a prompt prefill
/// and the per-token decode step that follows it.
///
/// The ragged variants are the ones a real batcher needs: sequences arrive with
/// different prompt lengths, so rows sit at different cache positions, and both
/// the attention window and the rotary positions must be taken per row.
/// </summary>
public sealed class BatchedDecodeTests
{
    private static ModelConfig Cfg => new()
    {
        VocabSize = 64,
        HiddenDim = 32,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 2,
        FfnDim = 64,
        MaxSeqLen = 64,
    };

    private const int Batch = 4;

    private static Transformer BuildModel()
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 4242);
        return ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
    }

    private static IKVCache[] BuildCaches(int batch) =>
        [.. Enumerable.Range(0, Cfg.NumLayers)
            .Select(_ => (IKVCache)new KVCache(batch, Cfg.NumKvHeads, Cfg.MaxSeqLen, Cfg.HeadDim))];

    private static int[] Prompt(int length, int seed)
    {
        var rng = new Random(seed);
        return [.. Enumerable.Range(0, length).Select(_ => rng.Next(Cfg.VocabSize))];
    }

    private static float[] Row(Tensor<float> logits, int row)
    {
        int vocab = logits.Shape[^1];
        long offset = (long)row * vocab;
        return [.. logits.Data[(int)offset..(int)(offset + vocab)]];
    }

    private static KVCacheRowView[] Views(IKVCache[] caches) =>
        [.. caches.Select(c => new KVCacheRowView(c))];

    private static void MapRows(KVCacheRowView[] views, params int[] rows)
    {
        foreach (var view in views) view.SetRows(rows);
    }

    /// <summary>
    /// Prefill: N different prompts in one [N, L] pass must equal N single-row
    /// passes, row for row. Different prompts matter — identical rows would pass
    /// even if the batched path broadcast one sequence to every row.
    /// </summary>
    [Fact]
    public void BatchedPrefill_MatchesPerSequencePrefill()
    {
        const int len = 6;
        int[][] prompts = [.. Enumerable.Range(0, Batch).Select(i => Prompt(len, 100 + i))];

        using var model = BuildModel();

        var batched = new int[Batch, len];
        for (int b = 0; b < Batch; b++)
            for (int t = 0; t < len; t++)
                batched[b, t] = prompts[b][t];

        using var ids = new Tensor<int>(Batch, len);
        for (int i = 0; i < ids.ElementCount; i++) ids.Data[i] = batched[i / len, i % len];

        var batchCaches = BuildCaches(Batch);
        using var batchLogits = model.ForwardLastLogits(ids, batchCaches, 0, null);

        for (int b = 0; b < Batch; b++)
        {
            using var one = new Tensor<int>(1, len);
            for (int t = 0; t < len; t++) one.Data[t] = prompts[b][t];

            var caches = BuildCaches(1);
            using var expected = model.ForwardLastLogits(one, caches, 0, null);

            var want = Row(expected, 0);
            var got = Row(batchLogits, b);
            for (int v = 0; v < want.Length; v++)
                Assert.Equal(want[v], got[v], 4);
        }
    }

    /// <summary>
    /// The step that runs per generated token. Same shape as prefill but one
    /// column, over caches that already hold the prompts — so this also pins that
    /// the batched pass advances every sequence's cache by exactly one position.
    /// </summary>
    [Fact]
    public void BatchedDecodeStep_MatchesPerSequenceDecodeStep()
    {
        const int len = 5;
        int[][] prompts = [.. Enumerable.Range(0, Batch).Select(i => Prompt(len, 200 + i))];
        int[] nextTokens = [.. Enumerable.Range(0, Batch).Select(i => Prompt(1, 300 + i)[0])];

        using var model = BuildModel();

        // Batched: prompt, then one decode step.
        using var batchedPrompt = new Tensor<int>(Batch, len);
        for (int b = 0; b < Batch; b++)
            for (int t = 0; t < len; t++)
                batchedPrompt.Data[b * len + t] = prompts[b][t];

        var batchCaches = BuildCaches(Batch);
        using (var _ = model.ForwardLastLogits(batchedPrompt, batchCaches, 0, null)) { }

        using var batchedStep = new Tensor<int>(Batch, 1);
        for (int b = 0; b < Batch; b++) batchedStep.Data[b] = nextTokens[b];

        using var batchLogits = model.ForwardLastLogits(batchedStep, batchCaches, len, null);

        // Per-sequence: same two steps on a single-row cache.
        for (int b = 0; b < Batch; b++)
        {
            using var one = new Tensor<int>(1, len);
            for (int t = 0; t < len; t++) one.Data[t] = prompts[b][t];

            var caches = BuildCaches(1);
            using (var _ = model.ForwardLastLogits(one, caches, 0, null)) { }

            using var step = new Tensor<int>(1, 1);
            step.Data[0] = nextTokens[b];
            using var expected = model.ForwardLastLogits(step, caches, len, null);

            var want = Row(expected, 0);
            var got = Row(batchLogits, b);
            for (int v = 0; v < want.Length; v++)
                Assert.Equal(want[v], got[v], 4);
        }
    }

    /// <summary>
    /// Real batched decode: every sequence is at a different position because
    /// the prompts differ in length. The cache must keep a length per row, and
    /// RoPE must be applied per row. A single shared position would rotate row
    /// 3's token as if it sat at row 0's position and still produce plausible
    /// floating-point noise, so this compares against per-sequence answers.
    /// </summary>
    [Fact]
    public void RaggedBatchedDecode_MatchesPerSequenceDecode()
    {
        int[] lengths = [3, 5, 7, 2];
        int[][] prompts = [.. lengths.Select((len, i) => Prompt(len, 400 + i))];
        int[] nextTokens = [.. Enumerable.Range(0, Batch).Select(i => Prompt(1, 500 + i)[0])];

        using var model = BuildModel();
        var caches = BuildCaches(Batch);
        var views = Views(caches);

        // Prefill one sequence at a time into its own slot.
        for (int b = 0; b < Batch; b++)
        {
            MapRows(views, b);

            using var prompt = new Tensor<int>(1, lengths[b]);
            for (int t = 0; t < lengths[b]; t++) prompt.Data[t] = prompts[b][t];
            using var _ = model.ForwardLastLogits(prompt, views, 0, null);
        }

        for (int b = 0; b < Batch; b++)
            Assert.Equal(lengths[b], caches[0].LengthForRow(b));

        // One batched step over rows sitting at four different positions.
        MapRows(views, 0, 1, 2, 3);

        using var step = new Tensor<int>(Batch, 1);
        for (int b = 0; b < Batch; b++) step.Data[b] = nextTokens[b];
        using var batchLogits = model.ForwardLastLogits(step, views, 0, null);

        for (int b = 0; b < Batch; b++)
        {
            var single = BuildCaches(1);
            using var prompt = new Tensor<int>(1, lengths[b]);
            for (int t = 0; t < lengths[b]; t++) prompt.Data[t] = prompts[b][t];
            using (var _ = model.ForwardLastLogits(prompt, single, 0, null)) { }

            using var one = new Tensor<int>(1, 1);
            one.Data[0] = nextTokens[b];
            using var expected = model.ForwardLastLogits(one, single, lengths[b], null);

            var want = Row(expected, 0);
            var got = Row(batchLogits, b);
            for (int v = 0; v < want.Length; v++)
                Assert.Equal(want[v], got[v], 4);

            foreach (var cache in single) cache.Dispose();
        }

        for (int b = 0; b < Batch; b++)
            Assert.Equal(lengths[b] + 1, caches[0].LengthForRow(b));

        foreach (var cache in caches) cache.Dispose();
    }

    /// <summary>
    /// Prefill a chunk after a cached prefix in one row while the other rows sit
    /// elsewhere: the chunk must land right after that row's own history, and
    /// only that row may advance.
    /// </summary>
    [Fact]
    public void RaggedChunkInOneRow_LeavesOtherRowsUntouched()
    {
        using var model = BuildModel();
        var caches = BuildCaches(Batch);
        var views = Views(caches);

        MapRows(views, 1);
        using (var first = new Tensor<int>(1, 2))
        {
            first.Data[0] = 5; first.Data[1] = 9;
            using var _ = model.ForwardLastLogits(first, views, 0, null);
        }
        Assert.Equal(2, caches[0].LengthForRow(1));
        Assert.Equal(0, caches[0].LengthForRow(0));

        MapRows(views, 1);
        using (var more = new Tensor<int>(1, 3))
        {
            more.Data[0] = 4; more.Data[1] = 6; more.Data[2] = 8;
            using var _ = model.ForwardLastLogits(more, views, 2, null);
        }

        Assert.Equal(5, caches[0].LengthForRow(1));
        Assert.Equal(0, caches[0].LengthForRow(0));
        Assert.Equal(0, caches[0].LengthForRow(2));
        Assert.Equal(0, caches[0].LengthForRow(3));

        // A row that moves must take its own state with it.
        foreach (var cache in caches) cache.MoveRow(1, 3);
        Assert.Equal(0, caches[0].LengthForRow(1));
        Assert.Equal(5, caches[0].LengthForRow(3));

        foreach (var cache in caches) cache.Dispose();
    }

    /// <summary>
    /// Same property for Multi-head Latent Attention, whose decode path absorbs
    /// <c>kv_b</c> into the query and reads one latent per cached position. Rows
    /// at different lengths must not share a position or a latent count.
    /// </summary>
    [Fact]
    public void RaggedMlaBatch_MatchesPerSequenceMlaDecode()
    {
        var cfg = new ModelConfig
        {
            VocabSize = 64,
            HiddenDim = 32,
            NumLayers = 2,
            NumHeads = 4,
            NumKvHeads = 1,
            FfnDim = 64,
            MaxSeqLen = 64,
            KvLoraRank = 4,
            RopeDim = 2,
            ValueLength = 4,
        };
        Assert.True(cfg.IsMla);

        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(cfg, sharpConfig);
        WeightInitializer.InitializeRandomly(weights, 777);
        // The latent RMS norm keeps its allocated zeros; ones are the neutral weight.
        foreach (var block in weights.Blocks) block.KvANormW?.Data.Fill(1f);

        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        int latentDim = cfg.KvLoraRank + cfg.MlaRopeHeadDim;

        IKVCache[] Caches(int rows) =>
            [.. Enumerable.Range(0, cfg.NumLayers).Select(_ => (IKVCache)new MlaKvCache(rows, cfg.MaxSeqLen, latentDim))];

        int[] lengths = [3, 5, 7];
        int[][] prompts = [.. lengths.Select((len, i) => Prompt(len, 800 + i))];
        int[] nextTokens = [3, 17, 41];

        var caches = Caches(3);
        var views = Views(caches);

        for (int b = 0; b < 3; b++)
        {
            MapRows(views, b);
            using var prompt = new Tensor<int>(1, lengths[b]);
            for (int t = 0; t < lengths[b]; t++) prompt.Data[t] = prompts[b][t];
            using var _ = model.ForwardLastLogits(prompt, views, 0, null);
        }

        for (int b = 0; b < 3; b++)
            Assert.Equal(lengths[b], caches[0].LengthForRow(b));

        MapRows(views, 0, 1, 2);
        using var step = new Tensor<int>(3, 1);
        for (int b = 0; b < 3; b++) step.Data[b] = nextTokens[b];
        using var batchLogits = model.ForwardLastLogits(step, views, 0, null);

        for (int b = 0; b < 3; b++)
        {
            var single = Caches(1);
            using var prompt = new Tensor<int>(1, lengths[b]);
            for (int t = 0; t < lengths[b]; t++) prompt.Data[t] = prompts[b][t];
            using (var _ = model.ForwardLastLogits(prompt, single, 0, null)) { }

            using var one = new Tensor<int>(1, 1);
            one.Data[0] = nextTokens[b];
            using var expected = model.ForwardLastLogits(one, single, lengths[b], null);

            var want = Row(expected, 0);
            var got = Row(batchLogits, b);
            for (int v = 0; v < want.Length; v++)
                Assert.Equal(want[v], got[v], 4);

            foreach (var cache in single) cache.Dispose();
        }

        foreach (var cache in caches) cache.Dispose();
    }
}
