using System.Diagnostics;

namespace SharpMind.Core.Storage;

/// <summary>
/// Measures a device's sequential read throughput by streaming a bounded slice of a file
/// with a large buffer and <see cref="FileOptions.SequentialScan"/>. The result fills the
/// storage fields of a <see cref="DeviceProfile"/>, which callers use to budget streaming
/// residency (a slow disk wants fewer pinned experts; a fast one can afford more).
/// </summary>
public static class SequentialReadBenchmark
{
    /// <summary>Default read buffer — one MiB keeps the syscall overhead negligible.</summary>
    public const int DefaultBufferSize = 1 << 20;

    /// <summary>Default cap on bytes read, so a benchmark on a huge model stays quick.</summary>
    public const long DefaultMaxBytes = 256L << 20;

    /// <summary>
    /// Streams up to <paramref name="maxBytes"/> (and at most the file's length) sequentially
    /// from <paramref name="path"/> and returns the measured throughput. A missing or empty
    /// file, or any I/O error, yields <see cref="DeviceProfile.Unknown"/> rather than throwing —
    /// the benchmark is diagnostic and must never fail a caller that is merely probing.
    /// </summary>
    public static DeviceProfile Measure(string path, long maxBytes = DefaultMaxBytes, int bufferSize = DefaultBufferSize)
    {
        if (string.IsNullOrWhiteSpace(path) || bufferSize <= 0 || maxBytes <= 0)
            return DeviceProfile.Unknown(path);

        try
        {
            long length = new FileInfo(path).Length;
            if (length <= 0) return DeviceProfile.Unknown(path);
            long target = Math.Min(length, maxBytes);

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                bufferSize, FileOptions.SequentialScan);
            var buffer = new byte[Math.Min(bufferSize, (int)Math.Min(target, int.MaxValue))];

            long read = 0;
            var sw = Stopwatch.StartNew();
            while (read < target)
            {
                int want = (int)Math.Min(buffer.Length, target - read);
                int n = stream.Read(buffer, 0, want);
                if (n <= 0) break;
                read += n;
            }
            sw.Stop();

            return Build(path, read, sw.Elapsed.TotalSeconds);
        }
        catch (IOException)
        {
            return DeviceProfile.Unknown(path);
        }
        catch (UnauthorizedAccessException)
        {
            return DeviceProfile.Unknown(path);
        }
    }

    /// <summary>Asynchronous variant of <see cref="Measure"/>.</summary>
    public static async Task<DeviceProfile> MeasureAsync(string path, long maxBytes = DefaultMaxBytes,
        int bufferSize = DefaultBufferSize, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(path) || bufferSize <= 0 || maxBytes <= 0)
            return DeviceProfile.Unknown(path);

        try
        {
            long length = new FileInfo(path).Length;
            if (length <= 0) return DeviceProfile.Unknown(path);
            long target = Math.Min(length, maxBytes);

            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                bufferSize, FileOptions.Asynchronous | FileOptions.SequentialScan);
            var buffer = new byte[Math.Min(bufferSize, (int)Math.Min(target, int.MaxValue))];

            long read = 0;
            var sw = Stopwatch.StartNew();
            while (read < target)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int want = (int)Math.Min(buffer.Length, target - read);
                int n = await stream.ReadAsync(buffer.AsMemory(0, want), cancellationToken).ConfigureAwait(false);
                if (n <= 0) break;
                read += n;
            }
            sw.Stop();

            return Build(path, read, sw.Elapsed.TotalSeconds);
        }
        catch (IOException)
        {
            return DeviceProfile.Unknown(path);
        }
        catch (UnauthorizedAccessException)
        {
            return DeviceProfile.Unknown(path);
        }
    }

    private static DeviceProfile Build(string path, long bytesRead, double elapsedSeconds)
    {
        if (bytesRead <= 0)
            return DeviceProfile.Unknown(path);

        double bps = elapsedSeconds > 0 ? bytesRead / elapsedSeconds : 0;
        return new DeviceProfile
        {
            Source = path,
            BytesRead = bytesRead,
            ElapsedSeconds = elapsedSeconds,
            SequentialReadBytesPerSecond = bps,
        };
    }
}
