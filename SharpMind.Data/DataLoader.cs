using System.Runtime.CompilerServices;
using System.Threading.Channels;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;

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

    public DataLoader(
        PipelineNode pipeline,
        Func<string, int[]> tokenise,
        IBatchStrategy batcher,
        int prefetchBuffer = 10)
    {
        ArgumentNullException.ThrowIfNull(pipeline);
        ArgumentNullException.ThrowIfNull(tokenise);
        ArgumentNullException.ThrowIfNull(batcher);
        if (prefetchBuffer <= 0) throw new ArgumentOutOfRangeException(nameof(prefetchBuffer));

        _pipeline = pipeline;
        _tokenise = tokenise;
        _batcher = batcher;
        _prefetchBuffer = prefetchBuffer;
    }

    /// <summary>
    /// Streams training batches from the configured pipeline with prefetching.
    /// The prefetcher runs in the background to ensure the GPU doesn't starve.
    /// </summary>
    public async IAsyncEnumerable<TrainingBatch> LoadAsync(
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        // Bounded channel provides backpressure: if the GPU is slow, the CPU stops prefetching.
        var channel = Channel.CreateBounded<TrainingBatch>(new BoundedChannelOptions(_prefetchBuffer)
        {
            FullMode = BoundedChannelFullMode.Wait,
            SingleReader = true,
            SingleWriter = true
        });

        // Background producer: feeds the channel from the batcher
        var producer = Task.Run(async () =>
        {
            try
            {
                await foreach (var batch in _batcher.BatchAsync(_pipeline.ReadAsync(cancellationToken), _tokenise, cancellationToken))
                {
                    await channel.Writer.WriteAsync(batch, cancellationToken);
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
        }, cancellationToken);

        // Consumer: reads from the channel
        while (await channel.Reader.WaitToReadAsync(cancellationToken))
        {
            while (channel.Reader.TryRead(out var batch))
            {
                yield return batch;
            }
        }

        // Ensure producer is cleaned up
        await producer;
    }

    /// <summary>
    /// Human-readable description of the full pipeline graph for diagnostics.
    /// </summary>
    public string Describe() => _pipeline.Describe();
}
