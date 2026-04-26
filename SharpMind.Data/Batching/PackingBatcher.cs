using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Data.Batching;

/// <summary>
/// Dynamic packing batcher — fills each row of the batch greedily with token
/// sequences, concatenating multiple short documents end-to-end until the row
/// is full. EOS tokens separate documents within a row.
///
/// This eliminates nearly all padding for datasets with variable-length
/// documents (web text, books) and is the default strategy used by LLaMA,
/// GPT, and most modern open-weight training runs.
///
/// Each row is independent — no document spans multiple rows.
/// Documents longer than <see cref="MaxSeqLen"/> are truncated.
///
/// The output batch shape is [<see cref="BatchSize"/>, <see cref="MaxSeqLen"/>].
/// </summary>
public sealed class PackingBatcher : IBatchStrategy
{
    private readonly int _batchSize;
    private readonly int _maxSeqLen;
    private readonly int _eosTokenId;
    private readonly int _padTokenId;

    /// <param name="batchSize">Number of rows per batch.</param>
    /// <param name="maxSeqLen">Number of tokens per row (sequence length).</param>
    /// <param name="eosTokenId">Token ID inserted between packed documents.</param>
    /// <param name="padTokenId">Token ID used to fill unused positions at row end.</param>
    public PackingBatcher(
        int batchSize,
        int maxSeqLen,
        int eosTokenId = 2,   // common default; override to match your tokenizer
        int padTokenId = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);

        _batchSize = batchSize;
        _maxSeqLen = maxSeqLen;
        _eosTokenId = eosTokenId;
        _padTokenId = padTokenId;
    }

    public async IAsyncEnumerable<TrainingBatch> BatchAsync(
        IAsyncEnumerable<string> documents,
        Func<string, int[]> tokenise,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Each row is a List<int> we fill greedily
        var rows = Enumerable.Range(0, _batchSize)
                                  .Select(_ => new List<int>(_maxSeqLen))
                                  .ToArray();
        int currentRow = 0;

        await foreach (string doc in documents.WithCancellation(cancellationToken))
        {
            int[] tokens = tokenise(doc);
            if (tokens.Length == 0) continue;

            // Truncate to maxSeqLen - 1 to leave room for EOS
            int take = Math.Min(tokens.Length, _maxSeqLen - 1);

            // Find a row with enough space; if current is full, advance
            while (currentRow < _batchSize && rows[currentRow].Count + take + 1 > _maxSeqLen)
                currentRow++;

            if (currentRow == _batchSize)
            {
                // All rows are full — emit batch and reset
                yield return BuildBatch(rows);
                foreach (var row in rows) row.Clear();
                currentRow = 0;
            }

            var target = rows[currentRow];
            for (int i = 0; i < take; i++)
                target.Add(tokens[i]);
            target.Add(_eosTokenId);
        }

        // Emit any partially filled batch (last batch may have padding)
        if (rows.Any(r => r.Count > 0))
            yield return BuildBatch(rows);
    }

    private TrainingBatch BuildBatch(List<int>[] rows)
    {
        var tokenIds = new Tensor<int>(_batchSize, _maxSeqLen);
        var labels = new Tensor<int>(_batchSize, _maxSeqLen);
        var attentionMask = new Tensor<float>(_batchSize, _maxSeqLen);

        // Label token IDs are -100 for padding (ignored in cross-entropy loss)
        const int ignoreLabelId = -100;

        int realTokenCount = 0;

        for (int b = 0; b < _batchSize; b++)
        {
            var row = rows[b];
            var tokSpan = tokenIds.RowSpan(b);
            var lblSpan = labels.RowSpan(b);
            var mskSpan = attentionMask.RowSpan(b);

            for (int i = 0; i < _maxSeqLen; i++)
            {
                if (i < row.Count)
                {
                    tokSpan[i] = row[i];
                    // Labels are the next token; last position gets ignore label
                    lblSpan[i] = i + 1 < row.Count ? row[i + 1] : ignoreLabelId;
                    mskSpan[i] = 1f;
                    realTokenCount++;
                }
                else
                {
                    tokSpan[i] = _padTokenId;
                    lblSpan[i] = ignoreLabelId;
                    mskSpan[i] = 0f;
                }
            }
        }

        return new TrainingBatch(tokenIds, labels, attentionMask, realTokenCount);
    }
}
