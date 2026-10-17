using System.Linq;
using SharpMind.Core.Storage;
using Xunit;

namespace SharpMind.Tests.Core;

/// <summary>
/// E4: the sequential-read benchmark that fills a <see cref="DeviceProfile"/>'s storage
/// fields. The measured throughput is host-dependent, so the assertions pin the deterministic
/// parts — bytes read, clamping, validity and the failure path — rather than any absolute speed.
/// </summary>
public sealed class SequentialReadBenchmarkTests : IDisposable
{
    private readonly TempDirectory _temp = new();

    public void Dispose() => _temp.Dispose();

    [Fact]
    public void Measure_ReadsTheWholeFile_AndReportsPositiveThroughput()
    {
        string path = _temp.Write("sample.bin", new string('x', 8192));

        DeviceProfile profile = SequentialReadBenchmark.Measure(path);

        Assert.True(profile.IsValid);
        Assert.Equal(path, profile.Source);
        Assert.Equal(8192, profile.BytesRead);
        Assert.True(profile.ElapsedSeconds > 0);
        Assert.True(profile.SequentialReadBytesPerSecond > 0);
        Assert.True(profile.SequentialReadMiBPerSecond > 0);
    }

    [Fact]
    public void Measure_ClampsToMaxBytes()
    {
        string path = _temp.Write("sample.bin", new string('x', 8192));

        DeviceProfile profile = SequentialReadBenchmark.Measure(path, maxBytes: 1024);

        Assert.True(profile.IsValid);
        Assert.Equal(1024, profile.BytesRead);
    }

    [Fact]
    public async Task MeasureAsync_ReadsTheWholeFile()
    {
        string path = _temp.Write("sample.bin", new string('x', 4096));

        DeviceProfile profile = await SequentialReadBenchmark.MeasureAsync(path, bufferSize: 512);

        Assert.True(profile.IsValid);
        Assert.Equal(4096, profile.BytesRead);
    }

    [Fact]
    public void Measure_MissingFile_IsUnknownNotThrowing()
    {
        string path = Path.Combine(_temp.Path, "does-not-exist.bin");

        DeviceProfile profile = SequentialReadBenchmark.Measure(path);

        Assert.False(profile.IsValid);
        Assert.Equal(0, profile.BytesRead);
        Assert.Equal(path, profile.Source);
    }

    [Fact]
    public void Measure_EmptyFile_IsUnknown()
    {
        string path = _temp.Write("empty.bin", "");

        Assert.False(SequentialReadBenchmark.Measure(path).IsValid);
    }

    [Fact]
    public void Unknown_HasNoMeasurement()
    {
        DeviceProfile profile = DeviceProfile.Unknown("z");
        Assert.False(profile.IsValid);
        Assert.Equal("z", profile.Source);
        Assert.Equal(0, profile.SequentialReadMiBPerSecond);
    }
}
