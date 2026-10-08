namespace SharpMind.Core.Memory;

/// <summary>
/// A thread-safe pool of exact-size <see cref="byte"/> arrays used to keep streaming
/// weight loads allocation-free. Layers rotate through the pool as the forward pass
/// advances: <see cref="Acquire"/> borrows a buffer of the tensor's byte count, and
/// <see cref="Release"/> returns it when the layer is unloaded, so the hot reload loop
/// never hits the GC in steady state (a streaming decode measured 300–400 MiB of managed
/// allocation per token before reuse; a single mis-sized buffer degrades to a fresh array).
///
/// Retention is bounded so streaming stays a low-memory mode: the pool keeps at most
/// <see cref="ResidentLayers"/> layers of the largest single tensor seen (the block loop
/// keeps roughly that many layers resident at once), evicting in first-in-first-out order
/// past its byte budget. Set <see cref="MaxRetainedBytes"/> to a concrete value to pin the
/// budget instead. This is the same acquire/release discipline MoE expert residency will
/// reuse as a cache keyed by expert index.
/// </summary>
public sealed class ByteBufferPool
{
    private const long Unlimited = long.MaxValue;
    private readonly Lock _lock = new();
    private readonly Dictionary<int, HashSet<byte[]>> _free = new();
    private readonly Queue<byte[]> _order = new();
    private long _bytes;
    private int _largest = -1;
    private long _hits;
    private long _misses;

    /// <summary>
    /// Explicit byte budget for retained buffers. Null (default) derives the budget from
    /// <see cref="ResidentLayers"/> and the largest tensor size observed.
    /// </summary>
    public long? MaxRetainedBytes { get; set; }

    /// <summary>Number of resident layers the pool should budget for (see class docs).</summary>
    public int ResidentLayers { get; set; } = 1;

    public byte[] Acquire(int byteCount)
    {
        if (byteCount < 0)
            throw new ArgumentOutOfRangeException(nameof(byteCount), "Negative buffer size requested.");
        lock (_lock)
        {
            // An acquired buffer may have been evicted from retention, a size may never be
            // backed yet, or a concurrent prefetch may have drained this size — track so
            // steering code can see how well the rotation reuses.
            if (_free.TryGetValue(byteCount, out var set) && set.Count > 0)
            {
                _hits++;
                // HashSet iteration order is arbitrary — any buffer of a given size serves.
                byte[] reusable = set.First();
                set.Remove(reusable);
                return reusable;
            }
            _misses++;
            if (byteCount > _largest) _largest = byteCount;
            return new byte[byteCount];
        }
    }

    public void Release(byte[]? buffer)
    {
        if (buffer is null) return;
        lock (_lock)
        {
            if (!_free.TryGetValue(buffer.Length, out var set))
                _free[buffer.Length] = set = [];
            set.Add(buffer);
            _bytes += buffer.Length;
            _order.Enqueue(buffer);

            long cap = EffectiveCap();
            if (cap < Unlimited)
            {
                int guard = _order.Count + 1; // never spin past entries we hold
                while (_bytes > cap && _order.Count > 0 && guard-- > 0) Evict();
            }
        }
    }

    /// <summary>Retained bytes currently held. For diagnostics/tests.</summary>
    public long RetainedBytes => _bytes;

    /// <summary>Number of distinct free arrays held. For diagnostics/tests.</summary>
    public int RetainedCount => _order.Count;

    /// <summary>Borrows that reused a pre-existing buffer. For diagnostics.</summary>
    public long Hits => _hits;

    /// <summary>Borrows that allocated a fresh array (pool cold or evicted). For diagnostics.</summary>
    public long Misses => _misses;

    private void Evict()
    {
        byte[] victim = _order.Dequeue();
        _bytes -= victim.Length;
        if (_free.TryGetValue(victim.Length, out var set))
        {
            set.Remove(victim);
            if (set.Count == 0) _free.Remove(victim.Length);
        }
    }

    private long EffectiveCap()
    {
        if (MaxRetainedBytes is { } cap && cap > 0) return cap;
        if (_largest <= 0) return Unlimited; // nothing learned yet: hold what we get
        return (long)_largest * (ResidentLayers + 2);
    }
}