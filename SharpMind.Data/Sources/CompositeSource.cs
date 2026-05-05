using System.Runtime.CompilerServices;
using System.Threading.Channels;

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
        // Create a buffer (channel) for each source to pre-fetch the next item in parallel
        var channels = _sources.Select(_ => Channel.CreateBounded<string>(new BoundedChannelOptions(1)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        })).ToArray();

        // Start producers for each source
        var tasks = _sources.Select(async (source, i) =>
        {
            try
            {
                await foreach (var doc in source.ReadAsync(cancellationToken))
                {
                    await channels[i].Writer.WriteAsync(doc, cancellationToken);
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                channels[i].Writer.TryComplete(ex);
            }
            finally
            {
                channels[i].Writer.TryComplete();
            }
        }).ToArray();

        // Background task to ensure all producers are awaited eventually
        _ = Task.WhenAll(tasks);

        var active = new bool[_sources.Length];
        Array.Fill(active, true);
        int remaining = _sources.Length;

        try
        {
            while (remaining > 0)
            {
                for (int i = 0; i < _sources.Length; i++)
                {
                    if (!active[i]) continue;

                    // Wait for the next item from this specific source (preserving order)
                    if (await channels[i].Reader.WaitToReadAsync(cancellationToken))
                    {
                        if (channels[i].Reader.TryRead(out var doc))
                        {
                            yield return doc;
                        }
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
            // Cleanup is handled by the producers completing their channels
        }
    }
}