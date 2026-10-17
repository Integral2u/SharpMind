using System.Runtime.CompilerServices;
using SharpMind.Data.Records;

namespace SharpMind.Data.Batching;

/// <summary>
/// B5: adapts an existing string <see cref="IBatchStrategy"/> (e.g.
/// <see cref="PackingBatcher"/>) to the record interface, tokenising each
/// record's <see cref="DataRecord.Text"/>. Records carry no mask this way, so the
/// result matches the string path exactly.
/// </summary>
public sealed class StringBatchStrategyAdapter(IBatchStrategy inner) : IRecordBatchStrategy
{
    private readonly IBatchStrategy _inner = inner ?? throw new ArgumentNullException(nameof(inner));

    public async IAsyncEnumerable<TrainingBatch> BatchAsync(
        IAsyncEnumerable<DataRecord> records,
        Func<DataRecord, TokenizedExample> tokenise,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(records);
        ArgumentNullException.ThrowIfNull(tokenise);

        await foreach (var batch in _inner.BatchAsync(
            Texts(records, cancellationToken),
            text => tokenise(new DataRecord { Text = text }).TokenIds.ToArray(),
            cancellationToken))
        {
            yield return batch;
        }
    }

    private static async IAsyncEnumerable<string> Texts(
        IAsyncEnumerable<DataRecord> records,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var record in records.WithCancellation(cancellationToken))
            yield return record.Text;
    }
}

/// <summary>Extensions for bridging batch strategies.</summary>
public static class RecordBatchStrategyAdapters
{
    /// <summary>Wraps a string batch strategy so it can consume records.</summary>
    public static IRecordBatchStrategy ToRecordStrategy(this IBatchStrategy strategy)
    {
        ArgumentNullException.ThrowIfNull(strategy);
        return new StringBatchStrategyAdapter(strategy);
    }
}
