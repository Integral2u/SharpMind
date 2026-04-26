using System.Runtime.CompilerServices;

namespace SharpMind.Data.Sources;
/// <summary>
/// Combines multiple <see cref="IDataSource"/> instances into one stream.
///
/// Two composition modes:
///   <see cref="CompositionMode.Concatenate"/> — exhausts each source in order.
///     Use when sources represent distinct corpora to be used sequentially.
///   <see cref="CompositionMode.RoundRobin"/> — takes one document from each
///     source in turn, cycling until all are exhausted.
///     Use to interleave corpora and prevent the model from seeing one corpus
///     in a long uninterrupted block.
/// </summary>
public sealed class CompositeSource : IDataSource
{
    private readonly IDataSource[] _sources;
    private readonly CompositionMode _mode;

    public enum CompositionMode { Concatenate, RoundRobin }

    public CompositeSource(IEnumerable<IDataSource> sources, CompositionMode mode = CompositionMode.Concatenate)
    {
        ArgumentNullException.ThrowIfNull(sources);
        _sources = [.. sources];
        _mode = mode;

        if (_sources.Length == 0)
            throw new ArgumentException("At least one source is required.", nameof(sources));
    }

    public long? EstimatedCount =>
        _sources.All(s => s.EstimatedCount.HasValue)
            ? _sources.Sum(s => s.EstimatedCount!.Value)
            : null;

    public string Description =>
        $"Composite({_sources.Length} sources, {_mode}): [{string.Join(", ", _sources.Select(s => s.Description))}]";

    public async IAsyncEnumerable<string> ReadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (_mode == CompositionMode.Concatenate)
        {
            foreach (var source in _sources)
            {
                await foreach (string doc in source.ReadAsync(cancellationToken))
                    yield return doc;
            }
        }
        else
        {
            await foreach (string doc in RoundRobinAsync(cancellationToken))
                yield return doc;
        }
    }

    public async ValueTask DisposeAsync()
    {
        foreach (var source in _sources)
            await source.DisposeAsync().ConfigureAwait(false);
    }

    // ── Round-robin interleaving ──────────────────────────────────────────

    private async IAsyncEnumerable<string> RoundRobinAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        // Materialise one enumerator per source and advance them in rotation
        var enumerators = _sources
            .Select(s => s.ReadAsync(cancellationToken).GetAsyncEnumerator(cancellationToken))
            .ToList();

        var active = new bool[enumerators.Count];
        Array.Fill(active, true);
        int remaining = enumerators.Count;

        try
        {
            while (remaining > 0)
            {
                for (int i = 0; i < enumerators.Count; i++)
                {
                    if (!active[i]) continue;
                    cancellationToken.ThrowIfCancellationRequested();

                    if (await enumerators[i].MoveNextAsync().ConfigureAwait(false))
                    {
                        yield return enumerators[i].Current;
                    }
                    else
                    {
                        active[i] = false;
                        remaining--;
                    }
                }
            }
        }
        finally
        {
            foreach (var e in enumerators)
                await e.DisposeAsync().ConfigureAwait(false);
        }
    }
}