using System.Runtime.CompilerServices;
using System.Threading.Channels;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;

namespace SharpMind.Data;

/// <summary>
/// Top-level entry point for the data pipeline.
/// Connects a <see cref="PipelineNode"/> to a tokeniser and a batch strategy,
/// and exposes the result as an <see cref="IAsyncEnumerable{TrainingBatch}"/>.
/// </summary>
public sealed class DataLoader
{
    private readonly PipelineNode _pipeline;
    private readonly Func<string, int[]> _tokenise;
    private readonly IBatchStrategy _batcher;
    private readonly int _prefetchBuffer;
    private readonly int? _maxBatches;

    public DataLoader(
        PipelineNode pipeline,
        Func<string, int[]> tokenise,
        IBatchStrategy batcher,
        int prefetchBuffer = 10,
        int? maxBatches = null)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(tokenise);
        ArgumentNullException.ThrowIfNull(batcher);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(prefetchBuffer);
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(maxBatches ?? 1, 0);

        _pipeline = pipeline;
        _tokenise = tokenise;
        _batcher = batcher;
        _prefetchBuffer = prefetchBuffer;
        _maxBatches = maxBatches;
    }

    /// <summary>
    /// Streams training batches from the configured pipeline with prefetching.
    /// The prefetcher runs in the background to ensure the GPU doesn't starve.
    /// </summary>
    /// <param name="maxBatches">
    /// Optional upper bound on the number of batches to emit. When set, the
    /// pipeline is re-enumerated (epoch-style) until the budget is reached so
    /// a small corpus can feed arbitrarily long training runs. When null, the
    /// constructor budget is used; if that is also null, the pipeline is
    /// consumed exactly once and the stream completes when it ends.
    /// </param>
    public async IAsyncEnumerable<TrainingBatch> LoadAsync(
        int? maxBatches = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        int? budget = maxBatches ?? _maxBatches;
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(budget ?? 1, 0);

        // Bounded channel provides backpressure: if the GPU is slow, the CPU stops prefetching.
        var channel = Channel.CreateBounded<TrainingBatch>(new BoundedChannelOptions(_prefetchBuffer)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });

        // Linked so that abandoning the enumeration (e.g. the caller breaks out of
        // its `await foreach`) can cancel the producer even when the caller's own
        // token is never cancelled — see the finally block below.
        using var producerCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);

        // Background producer: feeds the channel from the batcher, re-enumerating
        // the pipeline (epoch style) until the batch budget is satisfied.
        var producer = Task.Run(async () =>
        {
            try
            {
                int produced = 0;
                bool firstPass = true;
                while (firstPass || budget is { } b && produced < b)
                {
                    firstPass = false;
                    int passBatches = 0;
                    await foreach (var batch in _batcher.BatchAsync(_pipeline.ReadAsync(producerCts.Token), _tokenise, producerCts.Token))
                    {
                        if (budget is { } remaining && produced >= remaining)
                            break;
                        await channel.Writer.WriteAsync(batch, producerCts.Token);
                        produced++;
                        passBatches++;
                    }
                    // A pass that yields nothing (empty/filtered corpus) would loop
                    // forever against a budget — terminate instead.
                    if (passBatches == 0)
                        break;
                }
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                channel.Writer.TryComplete(ex);
            }
            finally
            {
                channel.Writer.TryComplete();
            }
        }, producerCts.Token);

        try
        {
            // Consumer: reads from the channel
            while (await channel.Reader.WaitToReadAsync(cancellationToken))
            {
                while (channel.Reader.TryRead(out var batch))
                {
                    yield return batch;
                }
            }
        }
        finally
        {
            // Runs on every exit path — normal completion, early disposal from a
            // caller breaking out of its `await foreach`, or an exception — not
            // just the happy path. Without this, an abandoned producer keeps
            // running in the background, still holding the corpus source open.
            producerCts.Cancel();
            try
            {
                await producer;
            }
            catch (OperationCanceledException) { }

            // The producer may have written batches the consumer never read (early
            // break). Drain and dispose them now rather than leaving their
            // NativeMemory-backed tensors alive until finalisation. Safe only after
            // awaiting the producer above — nothing can write to the channel anymore.
            while (channel.Reader.TryRead(out var abandoned))
                abandoned.Dispose();
        }
    }

    /// <summary>
    /// Human-readable description of the full pipeline graph for diagnostics.
    /// </summary>
    public string Describe() => _pipeline.Describe();
}
