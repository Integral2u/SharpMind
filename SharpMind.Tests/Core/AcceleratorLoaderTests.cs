using System.Reflection;
using SharpMind.Core;
using SharpMind.Core.Plugins;

namespace SharpMind.Tests.Core;

/// <summary>Discovered by <see cref="AcceleratorLoader"/> when the test assembly is scanned.</summary>
public sealed class FakeAcceleratorPlugin : IAcceleratorPlugin
{
    public string Name => "fake";
    public string Description => "test double";
    public IReadOnlyList<object> Capabilities { get; } = [new FakeOverrides()];

    public sealed class FakeOverrides : IMappingOverrides
    {
        public IReadOnlyDictionary<string, string> GetOverrides(SharpMindConfig config)
            => new Dictionary<string, string> { [SharpMindConfig.KeySoftmax] = "fake" };
    }
}

/// <summary>Must be skipped: no parameterless constructor.</summary>
public sealed class NoCtorAcceleratorPlugin(string name) : IAcceleratorPlugin
{
    public string Name => name;
    public string Description => "";
    public IReadOnlyList<object> Capabilities => [];
}

public sealed class AcceleratorLoaderTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    [Fact]
    public void Scan_FindsPublicPluginsWithParameterlessCtor_AndWarnsOnTheRest()
    {
        var plugins = new List<IAcceleratorPlugin>();
        var warnings = new List<string>();

        AcceleratorLoader.Scan(typeof(FakeAcceleratorPlugin).Assembly, plugins, warnings);

        Assert.Single(plugins);
        Assert.Equal("fake", plugins[0].Name);
        Assert.Contains(warnings, w => w.Contains(nameof(NoCtorAcceleratorPlugin)));
    }

    [Fact]
    public void Scan_SkipsDuplicateNames_WithWarning()
    {
        var plugins = new List<IAcceleratorPlugin>();
        var warnings = new List<string>();

        AcceleratorLoader.Scan(typeof(FakeAcceleratorPlugin).Assembly, plugins, warnings);
        AcceleratorLoader.Scan(typeof(FakeAcceleratorPlugin).Assembly, plugins, warnings);

        Assert.Single(plugins);
        Assert.Contains(warnings, w => w.Contains("'fake'") && w.Contains("duplicate"));
    }

    [Fact]
    public void Capabilities_FiltersByType()
    {
        var plugins = new List<IAcceleratorPlugin> { new FakeAcceleratorPlugin() };

        var overrides = plugins.Capabilities<IMappingOverrides>().ToList();

        Assert.Single(overrides);
        Assert.Equal("fake", overrides[0].GetOverrides(new SharpMindConfig())[SharpMindConfig.KeySoftmax]);
        Assert.Empty(plugins.Capabilities<IDisposable>());
    }

    [Fact]
    public void LoadFrom_MissingOrBlankDirectory_ReturnsEmptyWithoutWarnings()
    {
        var a = AcceleratorLoader.LoadFrom(null, out var wa);
        var b = AcceleratorLoader.LoadFrom(Path.Combine(_dir.Path, "nope"), out var wb);

        Assert.Empty(a); Assert.Empty(wa);
        Assert.Empty(b); Assert.Empty(wb);
    }

    [Fact]
    public void LoadFrom_ScansSubfolders_AndTurnsBadDllsIntoWarnings()
    {
        // A nested folder proves the scan is recursive; a garbage .dll proves one
        // bad file cannot abort the load.
        string nested = Path.Combine(_dir.Path, "vendor", "deep");
        Directory.CreateDirectory(nested);
        File.WriteAllText(Path.Combine(nested, "broken.dll"), "this is not a PE file");
        File.Copy(typeof(FakeAcceleratorPlugin).Assembly.Location, Path.Combine(nested, "SharpMind.Tests.dll"));

        var plugins = AcceleratorLoader.LoadFrom(_dir.Path, out var warnings);

        Assert.Contains(plugins, p => p.Name == "fake");
        Assert.Contains(warnings, w => w.Contains("broken.dll"));
    }
}
