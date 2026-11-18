using System.Collections.Concurrent;

namespace SharpMind.Server;

/// <summary>
/// Serialises access to one loaded model's <see cref="SharpMind.Model.Transformer"/>.
///
/// A transformer keeps its activations in instance fields, so two forward passes
/// that overlap corrupt each other's state. Chat sessions each own a generator
/// over the shared model, and the batched completion generator runs its own
/// scheduler thread over the same model; without a lease those engines can
/// overlap. Every engine therefore takes this lease around the work that touches
/// the model — a whole chat response for a session, one forward pass for the
/// batcher — which keeps the engines interleaved at a safe granularity instead
/// of serialising whole completions.
///
/// Leases are per model id, so two different models still run concurrently.
/// </summary>
public sealed class ModelLeaseService : IDisposable
{
    private readonly ConcurrentDictionary<string, SemaphoreSlim> _gates = new(StringComparer.OrdinalIgnoreCase);
    private bool _disposed;

    /// <summary>
    /// Waits for exclusive use of <paramref name="modelId"/>'s transformer.
    /// Dispose the returned lease to release it; disposing twice is safe.
    /// </summary>
    public async ValueTask<Lease> AcquireAsync(string modelId, CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, nameof(ModelLeaseService));

        var gate = _gates.GetOrAdd(modelId, static _ => new SemaphoreSlim(1, 1));
        await gate.WaitAsync(cancellationToken).ConfigureAwait(false);
        return new Lease(gate);
    }

    /// <summary>True when some engine currently holds the model (diagnostics only).</summary>
    public bool IsBusy(string modelId) =>
        _gates.TryGetValue(modelId, out var gate) && gate.CurrentCount == 0;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var gate in _gates.Values) gate.Dispose();
        _gates.Clear();
    }

    /// <summary>Holds the model; disposing twice releases once.</summary>
    public sealed class Lease(SemaphoreSlim gate) : IDisposable
    {
        private SemaphoreSlim? _gate = gate;

        public void Dispose()
        {
            // Exchange makes a double release a no-op instead of raising the count.
            Interlocked.Exchange(ref _gate, null)?.Release();
        }
    }
}
