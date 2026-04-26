namespace SharpMind.Data.Batching;
/// <summary>
/// Converts a stream of token ID sequences into <see cref="TrainingBatch"/> instances.
///
/// <paramref name="tokenise"/> is a delegate rather than a tokenizer interface
/// so that <c>SharpMind.Data</c> does not take a hard dependency on
/// <c>SharpMind.Tokenizer</c> — any callable that converts a string to
/// <c>int[]</c> works here.
/// </summary>
public interface IBatchStrategy
{
    /// <summary>
    /// Consumes the document stream produced by a pipeline node,
    /// tokenises each document, and yields training batches.
    /// </summary>
    IAsyncEnumerable<TrainingBatch> BatchAsync(
        IAsyncEnumerable<string> documents,
        Func<string, int[]> tokenise,
        CancellationToken cancellationToken = default);
}