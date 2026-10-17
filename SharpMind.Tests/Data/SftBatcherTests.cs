using SharpMind.Core.Tensors;
using SharpMind.Data.Batching;
using SharpMind.Data.Records;
using SharpMind.Training.Loss;

namespace SharpMind.Tests.Data;

/// <summary>
/// B5: <see cref="SftBatcher"/> masks the loss onto assistant tokens only. Every
/// label that predicts a non-assistant token (prompt, template scaffolding,
/// padding) is the ignore id, so a cross-entropy loss over the batch scores
/// exactly the assistant output.
/// </summary>
public sealed class SftBatcherTests
{
    private static DataRecord Rec(string text = "irrelevant") => new() { Text = text };

    private static async IAsyncEnumerable<DataRecord> Records(params DataRecord[] records)
    {
        foreach (var record in records)
        {
            await Task.Yield();
            yield return record;
        }
    }

    private static async IAsyncEnumerable<string> Strings(IEnumerable<string> items)
    {
        foreach (var item in items)
        {
            await Task.Yield();
            yield return item;
        }
    }

    private static async Task<List<TrainingBatch>> Batch(SftBatcher batcher, Func<DataRecord, TokenizedExample> tokenise, int count)
    {
        var batches = new List<TrainingBatch>();
        await foreach (var b in batcher.BatchAsync(Records(Enumerable.Range(0, count).Select(_ => Rec()).ToArray()), tokenise))
            batches.Add(b);
        return batches;
    }

    [Fact]
    public async Task NonAssistantTargets_GetTheIgnoreLabel()
    {
        var batcher = new SftBatcher(batchSize: 1, maxSeqLen: 8);
        // tokens 1..6; assistant output is tokens 4,5,6 (indices 3,4,5)
        var tokenise = (DataRecord _) => new TokenizedExample(
            [1, 2, 3, 4, 5, 6],
            [false, false, false, true, true, true]);

        var batch = (await Batch(batcher, tokenise, count: 1))[0];

        Assert.Equal(new[] { 1, 2, 3, 4, 5, 6, 0, 0 }, batch.TokenIds.Data.ToArray());
        Assert.Equal(new[] { -100, -100, 4, 5, 6, -100, -100, -100 }, batch.Labels.Data.ToArray());
        Assert.Equal(new[] { 1f, 1f, 1f, 1f, 1f, 1f, 0f, 0f }, batch.AttentionMask.Data.ToArray());
        Assert.Equal(6, batch.RealTokenCount);
    }

    [Fact]
    public async Task NullMask_MeansEveryTokenIsAssistant()
    {
        var batcher = new SftBatcher(batchSize: 1, maxSeqLen: 8);
        var tokenise = (DataRecord _) => new TokenizedExample([1, 2, 3, 4, 5, 6]);

        var batch = (await Batch(batcher, tokenise, count: 1))[0];

        // Standard causal shift with no masking: only the final position and the
        // padding are ignored.
        Assert.Equal(new[] { 2, 3, 4, 5, 6, -100, -100, -100 }, batch.Labels.Data.ToArray());
    }

    [Fact]
    public async Task ShortExamples_ArePadded_OnePerRow()
    {
        var batcher = new SftBatcher(batchSize: 2, maxSeqLen: 8);
        int call = 0;
        var tokenise = (DataRecord _) => Interlocked.Increment(ref call) == 1
            ? new TokenizedExample([10, 11, 12], null)
            : new TokenizedExample([20], null);

        var batch = (await Batch(batcher, tokenise, count: 2))[0];

        // Row 0: tokens 10,11,12 then pad. Row 1: token 20 then pad.
        Assert.Equal(new[] { 10, 11, 12, 0, 0, 0, 0, 0 }, batch.TokenIds.RowSpan(0).ToArray());
        Assert.Equal(new[] { 11, 12, -100, -100, -100, -100, -100, -100 }, batch.Labels.RowSpan(0).ToArray());
        Assert.Equal(new[] { 20, 0, 0, 0, 0, 0, 0, 0 }, batch.TokenIds.RowSpan(1).ToArray());
        Assert.Equal(new[] { -100, -100, -100, -100, -100, -100, -100, -100 }, batch.Labels.RowSpan(1).ToArray());
        Assert.Equal(4, batch.RealTokenCount);
    }

    [Fact]
    public async Task OverlongExamples_AreTruncatedToMaxSeqLen()
    {
        var batcher = new SftBatcher(batchSize: 1, maxSeqLen: 4);
        var tokenise = (DataRecord _) => new TokenizedExample([1, 2, 3, 4, 5, 6], null);

        var batch = (await Batch(batcher, tokenise, count: 1))[0];

        Assert.Equal(new[] { 1, 2, 3, 4 }, batch.TokenIds.Data.ToArray());
        Assert.Equal(new[] { 2, 3, 4, -100 }, batch.Labels.Data.ToArray());
        Assert.Equal(4, batch.RealTokenCount);
    }

    [Fact]
    public async Task EmptyExamples_AreSkipped()
    {
        var batcher = new SftBatcher(batchSize: 2, maxSeqLen: 4);
        int call = 0;
        var tokenise = (DataRecord _) => Interlocked.Increment(ref call) == 1
            ? new TokenizedExample([])
            : new TokenizedExample([7, 8], null);

        var batches = await Batch(batcher, tokenise, count: 2);

        Assert.Single(batches);
        Assert.Equal(new[] { 7, 8, 0, 0 }, batches[0].TokenIds.RowSpan(0).ToArray());
    }

    /// <summary>
    /// The masked batch is only useful if the loss really ignores the masked
    /// positions: cross-entropy over the batch must equal cross-entropy over just
    /// the assistant-target positions.
    /// </summary>
    [Fact]
    public async Task Loss_IsComputedOnlyOnAssistantTokens()
    {
        var batcher = new SftBatcher(batchSize: 2, maxSeqLen: 6);
        int call = 0;
        var tokenise = (DataRecord _) => Interlocked.Increment(ref call) == 1
            ? new TokenizedExample([1, 2, 3, 4, 5], [false, false, true, true, true])
            : new TokenizedExample([6, 7, 8], [false, true, true]);

        var batch = (await Batch(batcher, tokenise, count: 2))[0];

        int T = batch.TokenIds.Shape.ElementCount;
        int V = 12;

        // Deterministic, well-conditioned logits.
        using var logits = new Tensor<float>(T, V);
        for (int t = 0; t < T; t++)
            for (int v = 0; v < V; v++)
                logits.Data[t * V + v] = (float)Math.Sin(0.37 * (t + 1) * (v + 1));

        using var labels = Tensor<int>.From(batch.Labels.Data, T);

        var loss = new CrossEntropyLoss(ignoreId: -100);
        float masked = loss.Compute(logits, labels);

        // Reference: average NLL over exactly the non-ignored positions.
        double total = 0;
        int n = 0;
        for (int t = 0; t < T; t++)
        {
            int label = batch.Labels.Data[t];
            if (label == -100) continue;
            var row = logits.RowSpan(t);
            float max = float.NegativeInfinity;
            for (int v = 0; v < V; v++) if (row[v] > max) max = row[v];
            float sumExp = 0f;
            for (int v = 0; v < V; v++) sumExp += MathF.Exp(row[v] - max);
            total -= (row[label] - max) - MathF.Log(sumExp);
            n++;
        }
        float expected = (float)(total / n);

        Assert.True(n > 0, "the fixture must contain at least one assistant target");
        Assert.Equal(expected, masked, precision: 5);
    }

    [Fact]
    public async Task StringStrategyAdapter_MatchesTheStringBatcher()
    {
        var records = Enumerable.Range(0, 6)
            .Select(i => new DataRecord { Text = $"document number {i}" })
            .ToArray();

        Func<DataRecord, TokenizedExample> tokenise =
            r => new TokenizedExample(TestTokens.Encode(r.Text, 16));

        var adapted = new List<int[]>();
        await foreach (var b in new PackingBatcher(batchSize: 2, maxSeqLen: 8)
            .ToRecordStrategy()
            .BatchAsync(Records(records), tokenise))
        {
            foreach (int row in Enumerable.Range(0, 2))
                adapted.Add(b.TokenIds.RowSpan(row).ToArray());
        }

        var direct = new List<int[]>();
        await foreach (var b in new PackingBatcher(batchSize: 2, maxSeqLen: 8)
            .BatchAsync(Strings(records.Select(r => r.Text)),
                s => TestTokens.Encode(s, 16)))
        {
            foreach (int row in Enumerable.Range(0, 2))
                direct.Add(b.TokenIds.RowSpan(row).ToArray());
        }

        Assert.Equal(direct.Count, adapted.Count);
        for (int i = 0; i < direct.Count; i++)
            Assert.Equal(direct[i], adapted[i]);
    }
}
