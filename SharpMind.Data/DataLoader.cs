using System.Runtime.CompilerServices;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;

namespace SharpMind.Data;

/// <summary>
/// Top-level entry point for the data pipeline.
/// Connects a <see cref="PipelineNode"/> to a tokeniser and a batch strategy,
/// and exposes the result as an <see cref="IAsyncEnumerable{TrainingBatch}"/>.
///
/// Usage:
/// <code>
/// var loader = new DataLoader(
///     pipeline : CleaningPipeline.From(new JsonlSource("train.jsonl"))
///                                .Pipe(new NormaliseWhitespace())
///                                .Pipe(new MinLengthFilter(50)),
///     tokenise : text => tokenizer.Encode(text),
///     batcher  : new PackingBatcher(batchSize: 8, maxSeqLen: 2048));
///
/// await foreach (var batch in loader.LoadAsync())
/// {
///     // batch.TokenIds: [8, 2048]
///     // batch.Labels:   [8, 2048]
/// }
/// </code>
/// </summary>
public sealed class DataLoader
{
    private readonly PipelineNode _pipeline;
    private readonly Func<string, int[]> _tokenise;
    private readonly IBatchStrategy _batcher;

    public DataLoader(
        PipelineNode pipeline,
        Func<string, int[]> tokenise,
        IBatchStrategy batcher)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(tokenise);
        ArgumentNullException.ThrowIfNull(batcher);

        _pipeline = pipeline;
        _tokenise = tokenise;
        _batcher = batcher;
    }

    /// <summary>
    /// Streams training batches from the configured pipeline.
    /// Each batch is owned by the caller — dispose when done to release tensor memory.
    /// </summary>
    public IAsyncEnumerable<TrainingBatch> LoadAsync(
        CancellationToken cancellationToken = default)
        => _batcher.BatchAsync(_pipeline.ReadAsync(cancellationToken), _tokenise, cancellationToken);

    /// <summary>
    /// Human-readable description of the full pipeline graph for diagnostics.
    /// </summary>
    public string Describe() => _pipeline.Describe();
}
