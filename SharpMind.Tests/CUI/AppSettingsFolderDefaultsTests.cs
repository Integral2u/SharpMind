using SharpMind.CUI.App;

namespace SharpMind.Tests.CUI;

/// <summary>
/// The folder-defaulting decision in <see cref="AppSettings.Load"/>, tested as a pure
/// function: the resolver is injected so no test creates a directory or reads the real
/// user settings file.
/// </summary>
public sealed class AppSettingsFolderDefaultsTests
{
    private const string Resolved = @"X:\resolved\Plugins";
    private const string Missing = @"X:\definitely\not\here";

    private static readonly string Existing = Path.GetTempPath();

    /// <summary>Records what it was asked to resolve, so a test can assert it was not called.</summary>
    private sealed class Resolver
    {
        public List<string?> Calls { get; } = [];
        public string Resolve(string? initial) { Calls.Add(initial); return Resolved; }
    }

    [Fact]
    public void BlankToolsFolder_IsGivenTheDefault()
    {
        // The bug this covers: the guard tested ToolsFolder for blankness but checked and
        // assigned PluginsFolder, so ToolsFolder was never defaulted. A null ToolsFolder
        // means SessionLauncher loads no tool DLLs and PermissionGate adds no tools root.
        var r = new Resolver();
        var settings = new AppSettings { PluginsFolder = Existing, ToolsFolder = null };

        AppSettings.ApplyFolderDefaults(settings, r.Resolve);

        Assert.Equal(Resolved, settings.ToolsFolder);
    }

    [Fact]
    public void BlankToolsFolder_DoesNotDisturbAValidPluginsFolder()
    {
        var r = new Resolver();
        var settings = new AppSettings { PluginsFolder = Existing, ToolsFolder = "   " };

        AppSettings.ApplyFolderDefaults(settings, r.Resolve);

        Assert.Equal(Existing, settings.PluginsFolder);
        Assert.Single(r.Calls);      // resolved once, for ToolsFolder only
    }

    [Fact]
    public void MissingToolsFolder_IsGivenTheDefault()
    {
        var r = new Resolver();
        var settings = new AppSettings { PluginsFolder = Existing, ToolsFolder = Missing };

        AppSettings.ApplyFolderDefaults(settings, r.Resolve);

        Assert.Equal(Resolved, settings.ToolsFolder);
    }

    [Fact]
    public void BlankPluginsFolder_IsGivenTheDefault()
    {
        var r = new Resolver();
        var settings = new AppSettings { PluginsFolder = null, ToolsFolder = Existing };

        AppSettings.ApplyFolderDefaults(settings, r.Resolve);

        Assert.Equal(Resolved, settings.PluginsFolder);
        Assert.Equal(Existing, settings.ToolsFolder);
        Assert.Single(r.Calls);
    }

    [Fact]
    public void BothFoldersValid_AreLeftAlone()
    {
        var r = new Resolver();
        var settings = new AppSettings { PluginsFolder = Existing, ToolsFolder = Existing };

        AppSettings.ApplyFolderDefaults(settings, r.Resolve);

        Assert.Equal(Existing, settings.PluginsFolder);
        Assert.Equal(Existing, settings.ToolsFolder);
        Assert.Empty(r.Calls);
    }

    [Fact]
    public void EachFolderIsResolvedFromItsOwnCurrentValue()
    {
        // The resolver falls back to the value it was handed when it cannot create the
        // directory, so passing the wrong field's value would make that fallback restore
        // the wrong path.
        var r = new Resolver();
        var settings = new AppSettings { PluginsFolder = Missing, ToolsFolder = Missing + "-tools" };

        AppSettings.ApplyFolderDefaults(settings, r.Resolve);

        Assert.Equal([Missing, Missing + "-tools"], r.Calls);
    }
}
