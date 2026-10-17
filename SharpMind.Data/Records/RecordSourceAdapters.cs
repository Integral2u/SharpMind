using System.Runtime.CompilerServices;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;

namespace SharpMind.Data.Records;

/// <summary>
/// B4: adapters that bridge the string and record worlds, so existing
/// <see cref="IDataSource"/>s and <see cref="ICleaningStage"/>s keep working
/// unchanged when the pipeline is record-typed.
/// </summary>
public static class RecordSourceAdapters
{
    /// <summary>
    /// Lifts a string <see cref="IDataSource"/> into an <see cref="IRecordSource"/>:
    /// each document becomes a <see cref="DataRecord"/> whose <see cref="DataRecord.Text"/>
    /// is the document and whose <see cref="DataRecord.Source"/> is the source's description.
    /// </summary>
    public static IRecordSource ToRecordSource(this IDataSource source, string? sourceLabel = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        return new StringRecordSource(source, sourceLabel);
    }

    /// <summary>
    /// Projects an <see cref="IRecordSource"/> down to an <see cref="IDataSource"/>:
    /// each record contributes its <see cref="DataRecord.Text"/>. Structured extras
    /// (turns, tags) are not representable as a string and are dropped.
    /// </summary>
    public static IDataSource ToStringSource(this IRecordSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return new RecordStringSource(source);
    }

    /// <summary>
    /// Applies a string stage to each record's <see cref="DataRecord.Text"/>:
    /// a null result drops the record, otherwise the text is replaced and the
    /// remaining fields carried through. This is how existing cleaning stages run
    /// on records unchanged.
    /// </summary>
    public static IRecordSource Pipe(this IRecordSource source, ICleaningStage stage)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(stage);
        return new RecordTransformSource(source, stage.Name, record =>
        {
            string? text = stage.Process(record.Text);
            return text is null ? null : record with { Text = text };
        });
    }

    /// <summary>Applies an arbitrary record transform (null drops the record).</summary>
    public static IRecordSource Pipe(this IRecordSource source, string name, Func<DataRecord, DataRecord?> transform)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(transform);
        return new RecordTransformSource(source, name, transform);
    }

    /// <summary>Captures the records of a pipeline as a record source.</summary>
    public static IRecordSource AsRecordSource(this PipelineNode pipeline, string? sourceLabel = null)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        return new PipelineRecordSource(pipeline, sourceLabel);
    }
}

/// <summary>An <see cref="IRecordSource"/> that wraps a string <see cref="IDataSource"/>.</summary>
public sealed class StringRecordSource(IDataSource inner, string? sourceLabel = null) : IRecordSource
{
    private readonly IDataSource _inner = inner ?? throw new ArgumentNullException(nameof(inner));

    public long? EstimatedCount => _inner.EstimatedCount;

    public string Description => $"Records({_inner.Description})";

    public async IAsyncEnumerable<DataRecord> ReadRecordsAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (string doc in _inner.ReadAsync(cancellationToken))
            yield return new DataRecord { Text = doc, Source = sourceLabel ?? _inner.Description };
    }

    public ValueTask DisposeAsync() => _inner.DisposeAsync();
}

/// <summary>An <see cref="IDataSource"/> that exposes a record source's text.</summary>
public sealed class RecordStringSource(IRecordSource inner) : IDataSource
{
    private readonly IRecordSource _inner = inner ?? throw new ArgumentNullException(nameof(inner));

    public long? EstimatedCount => _inner.EstimatedCount;

    public string Description => $"Text({_inner.Description})";

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var record in _inner.ReadRecordsAsync(cancellationToken))
            yield return record.Text;
    }

    public ValueTask DisposeAsync() => _inner.DisposeAsync();
}

/// <summary>An <see cref="IRecordSource"/> that applies a transform to each record.</summary>
public sealed class RecordTransformSource(
    IRecordSource inner,
    string name,
    Func<DataRecord, DataRecord?> transform) : IRecordSource
{
    private readonly IRecordSource _inner = inner ?? throw new ArgumentNullException(nameof(inner));
    private readonly Func<DataRecord, DataRecord?> _transform = transform ?? throw new ArgumentNullException(nameof(transform));

    public long? EstimatedCount => _inner.EstimatedCount;

    public string Description => $"{_inner.Description} → {name}";

    public async IAsyncEnumerable<DataRecord> ReadRecordsAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var record in _inner.ReadRecordsAsync(cancellationToken))
        {
            var result = _transform(record);
            if (result is not null) yield return result;
        }
    }

    public ValueTask DisposeAsync() => _inner.DisposeAsync();
}

/// <summary>An <see cref="IRecordSource"/> that lifts a string <see cref="PipelineNode"/>.</summary>
public sealed class PipelineRecordSource(PipelineNode pipeline, string? sourceLabel = null) : IRecordSource
{
    private readonly PipelineNode _pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));

    public long? EstimatedCount => null;

    public string Description => $"Records({_pipeline.Describe().Split('\n')[0]})";

    public async IAsyncEnumerable<DataRecord> ReadRecordsAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (string doc in _pipeline.ReadAsync(cancellationToken))
            yield return new DataRecord { Text = doc, Source = sourceLabel };
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
