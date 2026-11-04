using SharpMind.Data.Batching;

namespace SharpMind.Data;

/// <summary>
/// A reusable source of training batches. <see cref="DataLoader"/> implements
/// this for string pipelines; record-aware loaders (for example SFT) implement
/// it without changing the existing DataLoader API.
/// </summary>
public interface ITrainingBatchSource
{
    /// <summary>
    /// Streams batches, optionally bounded by <paramref name="maxBatches"/> and
    /// skipping <paramref name="skipBatches"/> already-consumed batches.
    /// </summary>
    IAsyncEnumerable<TrainingBatch> LoadAsync(
        int? maxBatches = null,
        int skipBatches = 0,
        CancellationToken cancellationToken = default);
}
