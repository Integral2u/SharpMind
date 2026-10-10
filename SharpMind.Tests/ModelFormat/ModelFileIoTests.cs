using SharpMind.Model.Format;

namespace SharpMind.Tests.ModelFormat;

/// <summary>
/// Model loads retry past a transient Windows sharing/lock violation (an
/// antivirus or indexer short-term exclusive handle on a freshly written
/// .gguf/.smm). These tests pin that behavior — without the retry, opening
/// a file another handle owns is a hard IOException, which is what showed
/// up as a flaky CI failure on Windows agents.
/// </summary>
public sealed class ModelFileIoTests
{
    [Fact]
    public void OpenRead_RetriesPastAnExclusiveHandleAndSucceeds()
    {
        using var temp = new TempDirectory();
        string path = temp.Write("model.smm", "hello-model");
        using var blocker = new DelayedRelease(path, releaseAfterMs: 350);

        var watch = System.Diagnostics.Stopwatch.StartNew();
        using var stream = ModelFileIo.OpenRead(path);
        watch.Stop();

        using var reader = new StreamReader(stream);
        Assert.Equal("hello-model", reader.ReadToEnd());
        Assert.True(watch.ElapsedMilliseconds < 5000,
            $"retries should reach the file within the budget, took {watch.ElapsedMilliseconds}ms");
    }

    [Fact]
    public void OpenRead_MissingFile_ThrowsImmediately()
    {
        string path = System.IO.Path.Combine(
            System.IO.Path.GetTempPath(),
            $"missing-{Guid.NewGuid():N}.smm");

        var watch = System.Diagnostics.Stopwatch.StartNew();
        Assert.Throws<System.IO.FileNotFoundException>(() => ModelFileIo.OpenRead(path));
        watch.Stop();

        Assert.True(watch.ElapsedMilliseconds < 1000,
            "a missing file must not wait out the retry budget");
    }

    /// <summary>Holds an exclusive (FileShare.None) handle for <paramref name="releaseAfterMs"/>, then frees it.</summary>
    private sealed class DelayedRelease : IDisposable
    {
        private readonly System.Threading.Timer _timer;
        private readonly FileStream _stream;

        public DelayedRelease(string path, int releaseAfterMs)
        {
            _stream = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            _timer = new System.Threading.Timer(
                _ => _stream.Dispose(), null, releaseAfterMs, System.Threading.Timeout.Infinite);
        }

        public void Dispose()
        {
            _timer.Dispose();
            _stream.Dispose();
        }
    }
}