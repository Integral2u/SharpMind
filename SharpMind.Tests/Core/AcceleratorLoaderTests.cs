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

/// <summary>
/// Must be skipped with a warning, without aborting the scan of the rest of the
/// assembly: simulates a plugin whose <see cref="Name"/> getter touches driver
/// state and throws.
/// </summary>
public sealed class ThrowingNameAcceleratorPlugin : IAcceleratorPlugin
{
    public string Name => throw new InvalidOperationException("driver not found");
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

        Assert.Contains(plugins, p => p.Name == "fake");
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
    public void Scan_ThrowingNameGetter_WarnsAndKeepsScanningRestOfAssembly()
    {
        var plugins = new List<IAcceleratorPlugin>();
        var warnings = new List<string>();

        AcceleratorLoader.Scan(typeof(FakeAcceleratorPlugin).Assembly, plugins, warnings);

        // The healthy "fake" plugin, declared in the same assembly, still comes through —
        // proof that the throw did not abort the rest of the walk.
        Assert.Contains(plugins, p => p.Name == "fake");
        Assert.Contains(warnings, w => w.Contains(nameof(ThrowingNameAcceleratorPlugin)));
    }

    [Fact]
    public void HandleTypeLoadFailure_WarnsWithLoaderExceptionMessages_AndReturnsResolvedTypes()
    {
        // A genuine ReflectionTypeLoadException, built the same way the CLR would:
        // one type resolved, one slot null with a corresponding loader exception.
        var ex = new ReflectionTypeLoadException(
            classes: [typeof(FakeAcceleratorPlugin), null],
            exceptions: [new FileNotFoundException("Could not load file or assembly 'Vendor.Native, Version=1.0.0.0'")]);
        var warnings = new List<string>();

        Type[] resolved = AcceleratorLoader.HandleTypeLoadFailure(ex, typeof(FakeAcceleratorPlugin).Assembly, warnings);

        Assert.Equal([typeof(FakeAcceleratorPlugin)], resolved);
        Assert.Contains(warnings, w => w.Contains("Vendor.Native"));
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
        // A nested folder proves the scan is recursive; a locked file proves one
        // bad file cannot abort the load. This must be a genuine managed-load
        // failure, not BadImageFormatException — that case is a native dependency
        // beside the plugin and is covered (silently) by the test below.
        string nested = Path.Combine(_dir.Path, "vendor", "deep");
        Directory.CreateDirectory(nested);
        string lockedPath = Path.Combine(nested, "locked.dll");
        File.WriteAllText(lockedPath, "placeholder");
        File.Copy(typeof(FakeAcceleratorPlugin).Assembly.Location, Path.Combine(nested, "SharpMind.Tests.dll"));

        using var lockHandle = new FileStream(lockedPath, FileMode.Open, FileAccess.Read, FileShare.None);
        var plugins = AcceleratorLoader.LoadFrom(_dir.Path, out var warnings);

        Assert.Contains(plugins, p => p.Name == "fake");
        Assert.Contains(warnings, w => w.Contains("locked.dll"));
    }

    [Fact]
    public void LoadFrom_NativeDependencyBesidePlugin_ProducesNoWarning()
    {
        // A native DLL a plugin ships alongside itself (cudart64_12.dll style) throws
        // BadImageFormatException from Assembly.LoadFrom — expected, not a failure.
        File.WriteAllText(Path.Combine(_dir.Path, "cudart64_12.dll"), "this is not a PE file");
        File.Copy(typeof(FakeAcceleratorPlugin).Assembly.Location, Path.Combine(_dir.Path, "SharpMind.Tests.dll"));

        var plugins = AcceleratorLoader.LoadFrom(_dir.Path, out var warnings);

        Assert.Contains(plugins, p => p.Name == "fake");
        Assert.DoesNotContain(warnings, w => w.Contains("cudart64_12.dll"));
    }
}
