using SharpMind.Core.Plugins;
using SharpMind.GPU;
using SharpMind.Inference;
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
        var plugin = new IlgpuAcceleratorPlugin();

        Assert.Equal("ilgpu", plugin.Name);
        Assert.False(string.IsNullOrWhiteSpace(plugin.Description));
    }

    [Fact]
    public void Plugin_OffersATrainingEngineFactory()
    {
        // Exactly how TrainingEngineResolver locates the capability: filter Capabilities by type.
        var plugin = new IlgpuAcceleratorPlugin();
        Assert.Single(plugin.Capabilities.OfType<ITrainingEngineFactory>());
    }

    [Fact]
    public void Plugin_OffersAnInferenceEngineFactory()
    {
        // Exactly how InferenceEngineResolver locates the capability: filter Capabilities by type.
        // The inference capability itself must be a concrete GpuInferenceEngineFactory, and its
        // maximum prompt bound must be a sane default.
        var plugin = new IlgpuAcceleratorPlugin();
        var factory = Assert.Single(plugin.Capabilities.OfType<IInferenceEngineFactory>());
        var gpu = Assert.IsType<GpuInferenceEngineFactory>(factory);
        Assert.True(gpu.MaxPromptTokens > 0);
    }

    [Fact]
    public void Plugin_OffersABackendHintCapability()
    {
        // The options screen uses IBackendHintProvider (via the shared AcceleratorSelector) to
        // name the backend the ILGPU path would actually run — without the CUI referencing this
        // assembly. It must answer a sane value on any machine.
        var plugin = new IlgpuAcceleratorPlugin();
        var hint = Assert.Single(plugin.Capabilities.OfType<IBackendHintProvider>());
        string? value = hint.BackendHint();
        // null (only CPU fallback), OpenCL, or a CUDA/cuBLAS variant — never anything else.
        Assert.True(value is null
            || value == "OpenCL"
            || value == "CUDA"
            || value == "CUDA · cuBLAS");
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

        Assert.Contains(plugins, p => p.Name == "ilgpu");
        Assert.Empty(warnings);

        var ilgpu = plugins.First(p => p.Name == "ilgpu");
        Assert.Single(ilgpu.Capabilities.OfType<ITrainingEngineFactory>());
    }
}
