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
}
