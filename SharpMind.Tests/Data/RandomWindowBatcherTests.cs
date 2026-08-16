using SharpMind.Data.Batching;

namespace SharpMind.Tests.Data;

/// <summary>
/// Covers <see cref="RandomWindowBatcher"/>, the nanoGPT-style data feed:
/// batches are random contiguous <see cref="RandomWindowBatcher.SeqLen"/>-token
/// windows sliced from one flat corpus stream, labels are the next-token shift,
/// all positions are attended, and the flat buffer is built once and cached.
/// </summary>
public sealed class RandomWindowBatcherTests
{
    // Maps each ASCII digit character to its numeric token id; drops everything else.
    private static int[] Tokenise(string s) =>
        [.. s.Select(c => c - '0').Where(v => v is >= 0 and <= 9)];

    private static async IAsyncEnumerable<string> Source(params string[] docs)
    {
        await Task.CompletedTask;
        foreach (string doc in docs)
            yield return doc;
    }

    /// <summary>Enumerates one batch from <paramref name="batcher"/> and cancels.</summary>
    private static async Task<TrainingBatch> FirstBatch(RandomWindowBatcher batcher, Func<string, int[]>? tokenise = null)
    {
        Func<string, int[]> tokenizer = tokenise ?? Tokenise;
        using var cts = new CancellationTokenSource();
        await foreach (var batch in batcher.BatchAsync(Source("0123456789", "8765432109"), tokenizer)
                                .WithCancellation(cts.Token))
        {
            cts.Cancel();
            return batch;
        }
        throw new InvalidOperationException("No batch was produced.");
    }

    [Fact]
    public async Task Labels_AreNextTokenShift_AllPositionsMasked()
    {
        using var batch = await FirstBatch(new RandomWindowBatcher(batchSize: 2, seqLen: 4, seed: 7));

        Assert.Equal(2, batch.TokenIds.Shape.Rows);
        Assert.Equal(4, batch.TokenIds.Shape.Cols);
        Assert.Equal(2, batch.Labels.Shape.Rows);
        Assert.Equal(4, batch.Labels.Shape.Cols);
        Assert.Equal(2, batch.AttentionMask.Shape.Rows);
        Assert.Equal(4, batch.AttentionMask.Shape.Cols);

        for (int b = 0; b < 2; b++)
        {
            var tokens = batch.TokenIds.RowSpan(b).ToArray();
            var labels = batch.Labels.RowSpan(b).ToArray();
            var mask = batch.AttentionMask.RowSpan(b).ToArray();
            for (int s = 0; s < 4; s++)
            {
                // contiguous window → label[s] is the token at s+1 within the window
                if (s < 3)
                    Assert.Equal(tokens[s + 1], labels[s]);
                Assert.Equal(1f, mask[s]);
            }
        }
    }

    [Fact]
    public async Task SameSeed_ReproducesIdenticalWindows()
    {
        using var a = await FirstBatch(new RandomWindowBatcher(batchSize: 4, seqLen: 8, seed: 42));
        using var b = await FirstBatch(new RandomWindowBatcher(batchSize: 4, seqLen: 8, seed: 42));

        Assert.Equal(a.TokenIds.Data, b.TokenIds.Data);
        Assert.Equal(a.Labels.Data, b.Labels.Data);
    }

    [Fact]
    public async Task Buffer_IsCachedAcrossEnumerations()
    {
        int calls = 0;
        int[] CountingTokenise(string s) { calls++; return Tokenise(s); }

        var batcher = new RandomWindowBatcher(batchSize: 2, seqLen: 4, seed: 3);

        // First enumeration consumes the two documents → 2 tokenise calls.
        using var first = await FirstBatch(batcher, CountingTokenise);
        Assert.Equal(2, calls);

        // Second enumeration reuses the cached flat buffer — no re-read.
        using var second = await FirstBatch(batcher, CountingTokenise);
        Assert.Equal(2, calls);
    }

    [Fact]
    public async Task CorpusShorterThanWindow_Throws()
    {
        async Task Act()
        {
            using var cts = new CancellationTokenSource();
            var batcher = new RandomWindowBatcher(batchSize: 1, seqLen: 8, seed: 1);
            await foreach (var b in batcher.BatchAsync(Source("1234"), Tokenise).WithCancellation(cts.Token))
            {
                b.Dispose();
                break;
            }
        }

        // 4 tokens < 8+1 needed for a shifted window
        await Assert.ThrowsAsync<InvalidOperationException>(Act);
    }
}