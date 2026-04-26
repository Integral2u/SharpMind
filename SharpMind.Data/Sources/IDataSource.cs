namespace SharpMind.Data.Sources;

/// <summary>
/// A source of raw text documents streamed asynchronously.
///
/// Implementations read from files, JSONL datasets, byte streams, or any
/// other origin. Each yielded string is one logical document — a line,
/// a JSON record's text field, or an arbitrary chunk depending on the source.
///
/// Sources are intentionally dumb: they read and yield, nothing more.
/// Cleaning, filtering, and tokenisation happen downstream.
/// </summary>
public interface IDataSource : IAsyncDisposable
{
    /// <summary>
    /// Streams documents from this source.
    /// The caller controls cancellation via <paramref name="cancellationToken"/>.
    /// </summary>
    IAsyncEnumerable<string> ReadAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Estimated document count, if known without a full scan.
    /// Null when the source cannot determine this cheaply (e.g. a live stream).
    /// Used by the DataLoader to report progress.
    /// </summary>
    long? EstimatedCount { get; }

    /// <summary>Human-readable description for logging and diagnostics.</summary>
    string Description { get; }
}
