using SharpMind.Data.Batching;

namespace SharpMind.Data;

/// <summary>
/// B3: a fixed set of training batches materialised once and held in memory, so
/// a validation loop can score the exact same data on every step without
/// re-tokenising the corpus. Two evaluations over a <see cref="CachedBatchSet"/>
/// are therefore directly comparable.
///
/// The set owns the batches — including their native tensors — and yields them
/// borrowed. Consumers must NOT dispose the batches they receive; disposal is the
/// set's job (see <see cref="Dispose"/>). Callers that dispose as they go should
/// instead drain the loader themselves.
/// </summary>
public sealed class CachedBatchSet : IDisposable
{
    private readonly TrainingBatch[] _batches;

    public CachedBatchSet(IEnumerable<TrainingBatch> batches)
    {
        ArgumentNullException.ThrowIfNull(batches);
        _batches = [.. batches];
        TotalRealTokens = _batches.Sum(b => (long)b.RealTokenCount);
    }

    /// <summary>Number of cached batches.</summary>
    public int Count => _batches.Length;

    /// <summary>Total real (non-padding) tokens across every cached batch.</summary>
    public long TotalRealTokens { get; }

    /// <summary>
    /// The cached batches, borrowed. Do not dispose them — the set does. Stable
    /// across repeated evaluations.
    /// </summary>
    public IReadOnlyList<TrainingBatch> Batches => _batches;

    /// <summary>Disposes every cached batch and its tensors.</summary>
    public void Dispose()
    {
        foreach (var batch in _batches)
            batch.Dispose();
    }
}

/// <summary>Helpers for materialising batches into memory.</summary>
public static class DataLoaderCachingExtensions
{
    /// <summary>
    /// B3: reads exactly <paramref name="maxBatches"/> batches from
    /// <paramref name="loader"/> (re-enumerating the pipeline as needed) and
    /// returns them as a <see cref="CachedBatchSet"/>. If the loader's budget is
    /// smaller, only that many are cached.
    /// </summary>
    public static async Task<CachedBatchSet> LoadCachedAsync(
        this DataLoader loader,
        int maxBatches,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBatches);

        var list = new List<TrainingBatch>(maxBatches);
        await foreach (var batch in loader.LoadAsync(maxBatches: maxBatches, cancellationToken: cancellationToken))
            list.Add(batch);
        return new CachedBatchSet(list);
    }
}
