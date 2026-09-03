using SharpMind.Server;

namespace SharpMind.Tests.Server;

/// <summary>
/// Regression cover for the startup preload path
/// (<see cref="SharpMindService.PreloadModelAsync"/>).
///
/// <c>BuildHost</c> built the host and returned it without assigning
/// <c>_host</c>. <c>PreloadModelAsync</c> reads <c>_host</c>, so it threw
/// "Host not built." even though the CLI calls <c>BuildHost()</c> on the line
/// before — and <c>StartAsync</c>, whose <c>_host ??= BuildHost()</c> had no
/// cached value to find, then built a second host. The result was that
/// <c>--model</c> at startup never preloaded anything and reported a failure
/// while the server came up regardless.
///
/// These tests use an empty models directory: an unknown model id exercises
/// the whole path down to the directory scan without loading any weights.
/// </summary>
public sealed class ServiceStartupPreloadTests
{
    private static SharpMindServerOptions EmptyModelsDir(string dir) =>
        new() { ModelsDir = dir, Port = 0 };

    private static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), "sharpmind_test_" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    [Fact]
    public void BuildHost_IsIdempotent()
    {
        string dir = NewTempDir();
        try
        {
            var service = new SharpMindService(EmptyModelsDir(dir));

            var first = service.BuildHost();
            var second = service.BuildHost();

            // Not merely equal — the same host, or the service is holding a
            // different one than the caller was handed.
            Assert.Same(first, second);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task PreloadModelAsync_AfterBuildHost_DoesNotThrowHostNotBuilt()
    {
        string dir = NewTempDir();
        try
        {
            var service = new SharpMindService(EmptyModelsDir(dir));
            _ = service.BuildHost();   // exactly what the CLI does before preloading

            bool loaded = await service.PreloadModelAsync("absent.gguf");

            Assert.False(loaded);      // unknown id — but reached the scan, not an exception
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task PreloadModelAsync_WithoutBuildHost_BuildsItsOwnHost()
    {
        string dir = NewTempDir();
        try
        {
            var service = new SharpMindService(EmptyModelsDir(dir));

            bool loaded = await service.PreloadModelAsync("absent.gguf");

            Assert.False(loaded);
        }
        finally { Directory.Delete(dir, true); }
    }

    [Fact]
    public async Task PreloadModelAsync_UnknownModel_ReturnsFalse()
    {
        string dir = NewTempDir();
        try
        {
            // A real model file is present, so "not found" is about the id and
            // not about an empty directory.
            File.WriteAllBytes(Path.Combine(dir, "present.gguf"), [1, 2, 3]);
            var service = new SharpMindService(EmptyModelsDir(dir));

            Assert.False(await service.PreloadModelAsync("absent.gguf"));
        }
        finally { Directory.Delete(dir, true); }
    }
}
