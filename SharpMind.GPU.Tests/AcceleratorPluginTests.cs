using SharpMind.Core.Plugins;
using SharpMind.Training;

namespace SharpMind.GPU.Tests;

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
        // real DLL - not a reflection shortcut. SharpMind.GPU.dll sits in this test project's
        // output because it is a project reference, so this directory stands in for Plugins/.
        var plugins = AcceleratorLoader.LoadFrom(AppContext.BaseDirectory, out var warnings);

        Assert.Contains(plugins, p => p.Name == "cuda");
        Assert.Empty(warnings);

        var cuda = plugins.First(p => p.Name == "cuda");
        Assert.Single(cuda.Capabilities.OfType<ITrainingEngineFactory>());
    }
}
