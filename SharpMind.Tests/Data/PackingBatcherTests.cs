using SharpMind.Data.Batching;

namespace SharpMind.Tests.Data;

public sealed class PackingBatcherTests
{
    // Trivial tokeniser: splits on spaces, returns word indices from a fixed vocab
    private static int[] Tokenise(string s) =>
        TestTokens.Encode(s, 1000);

    private static async Task<List<TrainingBatch>> CollectBatches(
        IEnumerable<string> docs, PackingBatcher batcher)
    {
        async IAsyncEnumerable<string> Source()
        {
            await Task.CompletedTask;
            foreach (string d in docs) yield return d;
        }

        var batches = new List<TrainingBatch>();
        await foreach (var batch in batcher.BatchAsync(Source(), Tokenise))
            batches.Add(batch);
        return batches;
    }

    [Fact]
    public async Task BatchShape_IsCorrect()
    {
        var batcher = new PackingBatcher(batchSize: 2, maxSeqLen: 16);
        // Each doc tokenises to 3 tokens + EOS = 4 tokens; 4 docs fill 2 rows of 16
        var docs    = Enumerable.Repeat("a b c", 8).ToList();

        var batches = await CollectBatches(docs, batcher);

        Assert.True(batches.Count >= 1);
        var first = batches[0];
        Assert.Equal(2,  first.TokenIds.Shape.Rows);
        Assert.Equal(16, first.TokenIds.Shape.Cols);
        Assert.Equal(2,  first.Labels.Shape.Rows);
        Assert.Equal(16, first.Labels.Shape.Cols);
        Assert.Equal(2,  first.AttentionMask.Shape.Rows);
        Assert.Equal(16, first.AttentionMask.Shape.Cols);

        foreach (var b in batches) b.Dispose();
    }

    [Fact]
    public async Task AttentionMask_OneForRealTokens_ZeroForPadding()
    {
        // One doc of 3 tokens + EOS = 4 real tokens in a row of 16 → 12 padding
        var batcher = new PackingBatcher(batchSize: 1, maxSeqLen: 16, eosTokenId: 2, padTokenId: 0);
        var batches = await CollectBatches(["a b c"], batcher);

        Assert.Single(batches);
        var mask = batches[0].AttentionMask.RowSpan(0).ToArray();

        // First 4 positions real (3 tokens + EOS), rest padding
        Assert.All(mask.Take(4),    v => Assert.Equal(1f, v));
        Assert.All(mask.Skip(4),    v => Assert.Equal(0f, v));

        batches[0].Dispose();
    }

    [Fact]
    public async Task Labels_ShiftedByOne_PaddingIsMinusOneHundred()
    {
        var batcher = new PackingBatcher(batchSize: 1, maxSeqLen: 8, eosTokenId: 2, padTokenId: 0);
        // "a b" → 2 tokens + EOS = [tok_a, tok_b, 2, pad, pad, pad, pad, pad]
        var batches = await CollectBatches(["a b"], batcher);

        Assert.Single(batches);
        var tokens = batches[0].TokenIds.RowSpan(0).ToArray();
        var labels = batches[0].Labels.RowSpan(0).ToArray();

        // Label[i] = Token[i+1] for real tokens, -100 for last real + padding
        Assert.Equal(tokens[1], labels[0]);  // label for tok_a = tok_b
        Assert.Equal(2,         labels[1]);  // label for tok_b = EOS
        Assert.Equal(-100,      labels[2]);  // label for EOS = ignore
        Assert.All(labels.Skip(3), l => Assert.Equal(-100, l)); // padding = ignore

        batches[0].Dispose();
    }

    [Fact]
    public async Task RealTokenCount_ExcludesPadding()
    {
        var batcher = new PackingBatcher(batchSize: 1, maxSeqLen: 16);
        // 2 tokens + EOS = 3 real tokens
        var batches = await CollectBatches(["a b"], batcher);

        Assert.Single(batches);
        Assert.Equal(3, batches[0].RealTokenCount);

        batches[0].Dispose();
    }

    [Fact]
    public async Task PacksMultipleDocumentsPerRow()
    {
        // Each doc = 1 token + EOS = 2 tokens; maxSeqLen=8 can fit 4 docs per row
        var batcher = new PackingBatcher(batchSize: 1, maxSeqLen: 8, eosTokenId: 2);
        var docs    = Enumerable.Repeat("x", 4).ToList();
        var batches = await CollectBatches(docs, batcher);

        // All 4 docs packed into one row → 4 * 2 = 8 real tokens, one batch
        Assert.Single(batches);
        Assert.Equal(8, batches[0].RealTokenCount);

        batches[0].Dispose();
    }

    [Fact]
    public async Task DocumentLongerThanSeqLen_Truncated()
    {
        var batcher  = new PackingBatcher(batchSize: 1, maxSeqLen: 4);
        // 10 tokens + EOS would be 11 — must be truncated to 3 tokens + EOS = 4
        var longDoc  = string.Join(' ', Enumerable.Range(0, 10).Select(i => $"w{i}"));
        var batches  = await CollectBatches([longDoc], batcher);

        Assert.Single(batches);
        // Row should be fully utilised with no overflow
        Assert.Equal(4, batches[0].RealTokenCount);

        batches[0].Dispose();
    }

    [Fact]
    public async Task EmptyDocumentStream_YieldsNoBatches()
    {
        var batcher = new PackingBatcher(batchSize: 2, maxSeqLen: 16);
        var batches = await CollectBatches([], batcher);
        Assert.Empty(batches);
    }

    [Fact]
    public async Task PartialLastBatch_StillEmitted()
    {
        // batchSize=4 but only enough docs to fill 2 rows
        var batcher = new PackingBatcher(batchSize: 4, maxSeqLen: 8, eosTokenId: 2);
        // 2 docs, each fills one row of 8 (3 tokens + EOS + 3 tokens + EOS)
        var docs    = Enumerable.Repeat("a b c d e f g", 2).ToList();
        var batches = await CollectBatches(docs, batcher);

        // Should still get a batch even though only 2 of 4 rows are filled
        Assert.True(batches.Count >= 1);
        Assert.True(batches.Last().RealTokenCount > 0);

        foreach (var b in batches) b.Dispose();
    }

    [Fact]
    public async Task Dispose_ReleasesAllTensors()
    {
        var batcher = new PackingBatcher(batchSize: 1, maxSeqLen: 8);
        var batches = await CollectBatches(["hello world"], batcher);

        var ex = Record.Exception(() => { foreach (var b in batches) b.Dispose(); });
        Assert.Null(ex);
    }
}
