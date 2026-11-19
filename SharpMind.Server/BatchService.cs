using SharpMind.Inference;
using SharpMind.Model.Config;

namespace SharpMind.Server;

/// <summary>
/// Owns one <see cref="BatchedGenerator"/> per loaded model, so every completion
/// request for that model feeds the same batch instead of starting its own
/// single-sequence generator.
///
/// Lifetime is tied to <see cref="ModelManager"/>: a batcher is created on first
/// use and disposed when the model is unloaded. It holds no reference count of
/// its own — each request holds the model through
/// <see cref="ModelManager.LoadAsync"/> while it waits, so an unload cannot
/// dispose the transformer under an in-flight sequence.
/// </summary>
public sealed class BatchService : IDisposable
{
    private readonly ModelManager _models;
    private readonly SharpMindServerOptions _options;
    private readonly ModelLeaseService _leases;
    private readonly Lock _lock = new();
    private readonly Dictionary<string, Entry> _entries = new(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    public BatchService(ModelManager models, SharpMindServerOptions options, ModelLeaseService leases)
    {
        _models = models;
        _options = options;
        _leases = leases;
        _models.Unloaded += OnUnloaded;
    }

    /// <summary>
    /// The batcher for <paramref name="model"/>. The caller must hold the model
    /// (an outstanding <see cref="ModelManager.LoadAsync"/> reference) for as long
    /// as it uses the batcher.
    /// </summary>
    public BatchedGenerator Get(LoadedModel model)
    {
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, nameof(BatchService));

            if (_entries.TryGetValue(model.ModelId, out var existing))
            {
                if (ReferenceEquals(existing.Model.Model, model.Model))
                    return existing.Generator;

                // The model was reloaded under the same id; the old batcher's
                // transformer is gone, so it must not serve another request.
                _entries.Remove(model.ModelId);
                existing.Generator.Dispose();
            }

            var generator = Create(model);
            _entries[model.ModelId] = new Entry(model, generator);
            return generator;
        }
    }

    /// <summary>Aggregate counters over every live batcher, for /v1/health.</summary>
    public (int Models, long Steps, long Tokens, double TokensPerSecond, double AverageBatchSize) Snapshot()
    {
        lock (_lock)
        {
            long steps = 0, tokens = 0;
            double seconds = 0, batchTotal = 0;
            foreach (var entry in _entries.Values)
            {
                var stats = entry.Generator.Stats;
                steps += stats.Steps;
                tokens += stats.GeneratedTokens;
                seconds += stats.DecodeSeconds;
                batchTotal += stats.AverageBatchSize * stats.Steps;
            }
            return (_entries.Count, steps, tokens,
                seconds > 0 ? tokens / seconds : 0,
                steps > 0 ? batchTotal / steps : 0);
        }
    }

    private BatchedGenerator Create(LoadedModel model)
    {
        var config = model.Model.Config;
        int maxBatch = Math.Max(1, _options.CompletionBatchSize);

        // The KV memory budget is per process; ask for a length that fits
        // `maxBatch` rows rather than dividing a single row's length.
        int perRow = ModelConfig.ComputeMaxCacheLength(config, _options.MaxCacheLen, batchSize: maxBatch);

        var generator = new BatchedGenerator(model.Model, model.Tokenizer, new BatchedGeneratorOptions
        {
            MaxBatch = maxBatch,
            MaxSeqLen = perRow,
            MaxQueued = Math.Max(0, _options.MaxQueuedGenerations),
            AddBos = false,
            AddEos = false,
        });

        // Chat sessions drive the same transformer; the lease is held for each
        // forward pass only, so chat and batched completions interleave.
        generator.StepGate = ct => AcquireModelLeaseAsync(model.ModelId, ct);
        return generator;
    }

    private async ValueTask<IDisposable> AcquireModelLeaseAsync(string modelId, CancellationToken ct)
        => await _leases.AcquireAsync(modelId, ct).ConfigureAwait(false);

    private void OnUnloaded(string modelId)
    {
        Entry? entry = null;
        lock (_lock)
        {
            if (_entries.TryGetValue(modelId, out var found))
            {
                _entries.Remove(modelId);
                entry = found;
            }
        }

        // Disposal waits for the scheduler thread to stop. Nothing can be
        // in flight here: an unload requires the model's reference count to be
        // zero, and every request holds one while its sequence is queued.
        entry?.Generator.Dispose();
    }

    public void Dispose()
    {
        _models.Unloaded -= OnUnloaded;

        Entry[] entries;
        lock (_lock)
        {
            if (_disposed) return;
            _disposed = true;
            entries = [.. _entries.Values];
            _entries.Clear();
        }

        foreach (var entry in entries)
            entry.Generator.Dispose();
    }

    private sealed record Entry(LoadedModel Model, BatchedGenerator Generator);
}
