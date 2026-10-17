using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Data.Records;

namespace SharpMind.Data.Batching;

/// <summary>
/// B5: supervised fine-tuning batcher. One example per row — padded to
/// <see cref="MaxSeqLen"/>, no packing yet — with the label at every position set
/// to <see cref="IgnoreLabelId"/> unless the token it predicts is assistant output
/// (per the example's <see cref="TokenizedExample.AssistantMask"/>). Labels are
/// the standard causal shift: position <c>i</c> predicts token <c>i+1</c>.
///
/// The result is a batch whose cross-entropy loss is computed only on assistant
/// tokens — the user/system prompt, the template scaffolding and the padding all
/// contribute nothing — which is exactly what
/// <see cref="SharpMind.Training.Loss.CrossEntropyLoss"/> already supports via its
/// ignore id.
///
/// Output shape is [<see cref="BatchSize"/>, <see cref="MaxSeqLen"/>]. Packing
/// multiple examples into a row is a separate, later strategy; this one keeps one
/// example per row so masks never mix across examples.
/// </summary>
public sealed class SftBatcher : IRecordBatchStrategy
{
    /// <summary>Number of rows per batch.</summary>
    public int BatchSize { get; }

    /// <summary>Number of tokens per row (sequence length).</summary>
    public int MaxSeqLen { get; }

    /// <summary>Token ID used to pad unused positions at the end of a row.</summary>
    public int PadTokenId { get; }

    /// <summary>Label written where loss must be ignored (default <c>-100</c>).</summary>
    public int IgnoreLabelId { get; }

    public SftBatcher(int batchSize, int maxSeqLen, int padTokenId = 0, int ignoreLabelId = -100)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);

        BatchSize = batchSize;
        MaxSeqLen = maxSeqLen;
        PadTokenId = padTokenId;
        IgnoreLabelId = ignoreLabelId;
    }

    /// <inheritdoc/>
    public async IAsyncEnumerable<TrainingBatch> BatchAsync(
        IAsyncEnumerable<DataRecord> records,
        Func<DataRecord, TokenizedExample> tokenise,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);
        ArgumentNullException.ThrowIfNull(tokenise);

        var rows = new List<TokenizedExample>(BatchSize);

        await foreach (var record in records.WithCancellation(cancellationToken))
        {
            var example = tokenise(record);
            if (example.TokenIds.Count == 0) continue;

            rows.Add(example);
            if (rows.Count == BatchSize)
            {
                yield return BuildBatch(rows);
                rows.Clear();
            }
        }

        if (rows.Count > 0)
            yield return BuildBatch(rows);
    }

    private TrainingBatch BuildBatch(List<TokenizedExample> rows)
    {
        var tokenIds = new Tensor<int>(BatchSize, MaxSeqLen);
        var labels = new Tensor<int>(BatchSize, MaxSeqLen);
        var attentionMask = new Tensor<float>(BatchSize, MaxSeqLen);

        int realTokenCount = 0;

        for (int b = 0; b < BatchSize; b++)
        {
            var tokSpan = tokenIds.RowSpan(b);
            var lblSpan = labels.RowSpan(b);
            var mskSpan = attentionMask.RowSpan(b);

            if (b < rows.Count)
            {
                var example = rows[b];
                int len = Math.Min(example.TokenIds.Count, MaxSeqLen);

                for (int i = 0; i < MaxSeqLen; i++)
                {
                    if (i < len)
                    {
                        tokSpan[i] = example.TokenIds[i];
                        mskSpan[i] = 1f;
                        realTokenCount++;

                        // Causal shift: position i predicts token i+1, and only
                        // counts toward loss when that token is assistant output.
                        lblSpan[i] = i + 1 < len && example.IsAssistant(i + 1)
                            ? example.TokenIds[i + 1]
                            : IgnoreLabelId;
                    }
                    else
                    {
                        tokSpan[i] = PadTokenId;
                        mskSpan[i] = 0f;
                        lblSpan[i] = IgnoreLabelId;
                    }
                }
            }
            else
            {
                for (int i = 0; i < MaxSeqLen; i++)
                {
                    tokSpan[i] = PadTokenId;
                    mskSpan[i] = 0f;
                    lblSpan[i] = IgnoreLabelId;
                }
            }
        }

        return new TrainingBatch(tokenIds, labels, attentionMask, realTokenCount);
    }
}
