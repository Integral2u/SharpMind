using SharpMind.Core.Plugins;
using SharpMind.GPU;
using SharpMind.Training;

namespace SharpMind.Tests.GPU;

/// <summary>
/// The plugin surface the host binds to. These are deliberately device-independent: the point is
/// that discovery, capability lookup and the refusal path all behave on a machine with no GPU at
/// all, which is what upstream CI is.
/// </summary>
public sealed class AcceleratorPluginTests
{
    [Fact]
    public void Plugin_ConstructsWithoutTouchingTheDevice()
    {
        // IAcceleratorPlugin's lifetime contract: a parameterless constructor that is cheap and
        // side-effect free, because the host builds one on every plugins-folder scan (the training
        // wizard scans each time it opens) and never disposes it. Constructing must therefore work
        // with no driver present.
        var plugin = new CudaAcceleratorPlugin();

        Assert.Equal("cuda", plugin.Name);
        Assert.False(string.IsNullOrWhiteSpace(plugin.Description));
    }

    [Fact]
    public void Plugin_OffersATrainingEngineFactory()
    {
        // Exactly how TrainingEngineResolver locates the capability: filter Capabilities by type.
        var plugin = new CudaAcceleratorPlugin();
        Assert.Single(plugin.Capabilities.OfType<ITrainingEngineFactory>());
    }

    [Fact]
    public void Plugin_IsDiscoverableThroughTheHostsOwnLoader()
    {
        // Through the public entry point the CUI itself calls, against a real folder holding a
        // real DLL - not a reflection shortcut. The standalone GPU.Tests output contained only
        // SharpMind.GPU.dll and its dependency DLLs, so that folder stood in for Plugins/ with no
        // competing plugin types. Now that these tests live in the main suite, AppContext's output
        // also holds SharpMind.Tests.dll with its own IAcceleratorPlugin test fakes - so isolate the
        // scan folder to just the plugin and its dependencies, reproducing the original layout.
        using var dir = new TempDirectory();
        foreach (var dll in Directory.EnumerateFiles(AppContext.BaseDirectory, "*.dll"))
        {
            if (Path.GetFileName(dll) == "SharpMind.Tests.dll") continue;
            File.Copy(dll, Path.Combine(dir.Path, Path.GetFileName(dll)));
        }

        var plugins = AcceleratorLoader.LoadFrom(dir.Path, out var warnings);

        Assert.Contains(plugins, p => p.Name == "cuda");
        Assert.Empty(warnings);

        var cuda = plugins.First(p => p.Name == "cuda");
        Assert.Single(cuda.Capabilities.OfType<ITrainingEngineFactory>());
    }
}
