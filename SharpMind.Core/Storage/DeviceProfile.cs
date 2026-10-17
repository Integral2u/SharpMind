namespace SharpMind.Core.Storage;

/// <summary>
/// Measured storage characteristics of a device/path. Populated by
/// <see cref="SequentialReadBenchmark"/> and used to size streaming residency budgets
/// (e.g. how many MoE experts can be pinned given the media's read throughput).
/// </summary>
public sealed record DeviceProfile
{
    /// <summary>Path/host that was measured. Diagnostic only.</summary>
    public string? Source { get; init; }

    /// <summary>Bytes actually read during the measurement.</summary>
    public long BytesRead { get; init; }

    /// <summary>Wall-clock time the sequential read took, in seconds.</summary>
    public double ElapsedSeconds { get; init; }

    /// <summary>Sequential read throughput in bytes/second.</summary>
    public double SequentialReadBytesPerSecond { get; init; }

    /// <summary>Sequential read throughput in MiB/second (convenience unit).</summary>
    public double SequentialReadMiBPerSecond => SequentialReadBytesPerSecond / (1024.0 * 1024.0);

    /// <summary>True when the profile carries a real positive measurement.</summary>
    public bool IsValid => BytesRead > 0 && ElapsedSeconds > 0 && SequentialReadBytesPerSecond > 0;

    /// <summary>An empty profile for a path that could not be measured.</summary>
    public static DeviceProfile Unknown(string? source = null) => new() { Source = source };
}
