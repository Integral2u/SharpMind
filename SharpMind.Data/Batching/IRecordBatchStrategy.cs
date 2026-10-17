using SharpMind.Data.Records;

namespace SharpMind.Data.Batching;

/// <summary>
/// B5: the record-aware counterpart of <see cref="IBatchStrategy"/>. Where
/// <see cref="IBatchStrategy"/> takes strings and a <c>Func&lt;string, int[]&gt;</c>
/// — with no place to carry a per-token loss mask — this takes
/// <see cref="DataRecord"/>s and a tokeniser returning a
/// <see cref="TokenizedExample"/>, so a batcher can zero the loss on non-assistant
/// tokens (see <see cref="SftBatcher"/>).
///
/// Like <see cref="IBatchStrategy"/> the tokeniser is a delegate, keeping
/// <c>SharpMind.Data</c> free of a tokenizer or template dependency.
/// </summary>
public interface IRecordBatchStrategy
{
    /// <summary>
    /// Consumes a record stream, tokenises each record, and yields training
    /// batches (typically with masked labels).
    /// </summary>
    IAsyncEnumerable<TrainingBatch> BatchAsync(
        IAsyncEnumerable<DataRecord> records,
        Func<DataRecord, TokenizedExample> tokenise,
        CancellationToken cancellationToken = default);
}
