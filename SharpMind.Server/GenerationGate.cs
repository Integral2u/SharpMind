namespace SharpMind.Server;

/// <summary>
/// Raised when a generation request arrives while the queue is already at its
/// limit. The caller should be refused (HTTP 429) rather than parked: an
/// unbounded queue turns overload into a timeout for everyone.
/// </summary>
public sealed class GenerationQueueFullException(int maxConcurrent, int maxQueued)
    : Exception($"Server is at capacity ({maxConcurrent} generating, {maxQueued} queued); retry shortly.")
{
    public int MaxConcurrent { get; } = maxConcurrent;
    public int MaxQueued { get; } = maxQueued;
}

/// <summary>
/// Hard admission control for generation work.
///
/// Decoding is not free concurrency: every admitted request owns a KV cache and
/// a native workspace, and on a CPU the kernels already saturate the cores, so
/// admitting more generations than the machine can run in parallel converts a
/// latency problem into a memory problem. This gate admits at most
/// <see cref="MaxConcurrent"/> at a time and lets at most <see cref="MaxQueued"/>
/// wait behind them; anything beyond that is refused immediately.
///
/// It is deliberately not a rate limiter: requests that fit are never dropped,
/// and a queued request keeps its place until its turn or its cancellation.
/// </summary>
public sealed class GenerationGate : IDisposable
{
    private readonly SemaphoreSlim _slots;
    private readonly SemaphoreSlim? _queueSlots;
    private int _active;
    private int _waiting;

    /// <summary>
    /// A gate admitting <paramref name="maxConcurrent"/> generations with
    /// <paramref name="maxQueued"/> waiters behind them. A queue of zero makes
    /// every contended request fail fast, which is the right behaviour when the
    /// caller (not the server) is responsible for retrying.
    /// </summary>
    public GenerationGate(int maxConcurrent, int maxQueued)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxConcurrent, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxQueued, 0);

        MaxConcurrent = maxConcurrent;
        MaxQueued = maxQueued;
        _slots = new SemaphoreSlim(maxConcurrent, maxConcurrent);
        // Counts only requests that are waiting for a slot, so capacity is not
        // consumed by requests that are already generating. Null means "no
        // queue": a contended request is refused rather than parked.
        _queueSlots = maxQueued > 0 ? new SemaphoreSlim(maxQueued, maxQueued) : null;
    }

    public int MaxConcurrent { get; }
    public int MaxQueued { get; }

    /// <summary>Generations currently holding a slot.</summary>
    public int Active => Volatile.Read(ref _active);

    /// <summary>Requests waiting for a slot.</summary>
    public int Queued => Volatile.Read(ref _waiting);

    /// <summary>
    /// Waits for a generation slot and returns a lease that releases it. Throws
    /// <see cref="GenerationQueueFullException"/> without waiting when the queue
    /// is full, and <see cref="OperationCanceledException"/> when
    /// <paramref name="ct"/> fires while queued. Neither case consumes a slot.
    /// </summary>
    public async Task<Lease> EnterAsync(CancellationToken ct = default)
    {
        if (_queueSlots is null)
        {
            // No queue configured: take a slot if one is free, otherwise refuse.
            if (!_slots.Wait(0))
                throw new GenerationQueueFullException(MaxConcurrent, MaxQueued);
            Interlocked.Increment(ref _active);
            return new Lease(this);
        }

        if (!_queueSlots.Wait(0))
            throw new GenerationQueueFullException(MaxConcurrent, MaxQueued);

        Interlocked.Increment(ref _waiting);
        try
        {
            await _slots.WaitAsync(ct).ConfigureAwait(false);
        }
        catch
        {
            // Cancelled (or the wait faulted) while queued: give the queue place
            // back so a later request can wait, then rethrow.
            _queueSlots.Release();
            throw;
        }
        finally
        {
            Interlocked.Decrement(ref _waiting);
        }

        _queueSlots.Release();
        Interlocked.Increment(ref _active);
        return new Lease(this);
    }

    /// <summary>
    /// Takes a slot only if one is free right now. Used where waiting would be
    /// worse than refusing, e.g. a health probe that must not queue.
    /// </summary>
    public bool TryEnter(out Lease lease)
    {
        if (!_slots.Wait(0))
        {
            lease = null!;
            return false;
        }

        Interlocked.Increment(ref _active);
        lease = new Lease(this);
        return true;
    }

    private void Release()
    {
        Interlocked.Decrement(ref _active);
        _slots.Release();
    }

    public void Dispose()
    {
        _slots.Dispose();
        _queueSlots?.Dispose();
    }

    /// <summary>
    /// Holds one slot for the duration of a generation. Disposing twice is safe
    /// and releases once: an over-release would quietly raise the cap.
    /// </summary>
    public sealed class Lease(GenerationGate gate) : IDisposable
    {
        private GenerationGate? _gate = gate;

        public void Dispose()
        {
            // Exchange makes double-dispose a no-op instead of a second Release.
            Interlocked.Exchange(ref _gate, null)?.Release();
        }
    }
}
