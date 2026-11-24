namespace SharpMind.Core.Quantization;

/// <summary>
/// Process-wide accounting for repacked weight copies. A byte repack can be
/// several times the raw tensor, and many layers reserve at once, so a per-layer
/// cap is not enough: this counter makes <see cref="MaxBytes"/> a real aggregate
/// limit. Reservations are released by the owning wide-kernel instance.
/// </summary>
public static class WideRepackBudget
{
    private static long _allocated;

    /// <summary>Total repack bytes allowed across all live layers. Default 4 GiB.</summary>
    public static long MaxBytes { get; set; } = 4L << 30;

    public static long AllocatedBytes => Interlocked.Read(ref _allocated);

    /// <summary>Reserves <paramref name="bytes"/>, or returns false when the total would exceed the cap.</summary>
    public static bool TryReserve(long bytes)
    {
        if (bytes <= 0) return true;
        long current;
        do
        {
            current = Interlocked.Read(ref _allocated);
            if (current > MaxBytes - bytes) return false;
        }
        while (Interlocked.CompareExchange(ref _allocated, current + bytes, current) != current);
        return true;
    }

    public static void Release(long bytes)
    {
        if (bytes <= 0) return;
        Interlocked.Add(ref _allocated, -bytes);
    }

    /// <summary>Test hook: clears the counter without touching live instances.</summary>
    public static void ResetForTests() => Interlocked.Exchange(ref _allocated, 0);
}
