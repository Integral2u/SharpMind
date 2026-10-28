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

        // Hold an exclusive handle, then release it during the second backoff.
        // No wall-clock timing: the retry count is exact, so the test cannot
        // flake under load.
        using var blocker = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        int backoffs = 0;
        using var stream = ModelFileIo.OpenRead(path, sleep: _ =>
        {
            if (++backoffs == 2) blocker.Dispose();
        });

        using var reader = new StreamReader(stream);
        Assert.Equal("hello-model", reader.ReadToEnd());
        Assert.Equal(2, backoffs);
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
}