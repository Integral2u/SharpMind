namespace SharpMind.Data.Records;

/// <summary>
/// B4: the typed counterpart of <see cref="Sources.IDataSource"/> — streams
/// <see cref="DataRecord"/>s instead of strings. It exists in parallel with the
/// string path, not in place of it: sources and stages flow between the two
/// through the adapters in <see cref="RecordSourceAdapters"/>.
/// </summary>
public interface IRecordSource : IAsyncDisposable
{
    /// <summary>Estimated record count, if known without a full scan.</summary>
    long? EstimatedCount { get; }

    /// <summary>Human-readable description for logging and diagnostics.</summary>
    string Description { get; }

    /// <summary>Streams records from this source.</summary>
    IAsyncEnumerable<DataRecord> ReadRecordsAsync(CancellationToken cancellationToken = default);
}
