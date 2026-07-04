using System.Text.Json;
using System.Text.Json.Serialization;

namespace SharpMind.CUI.App;

/// <summary>
/// Settings that persist across runs of the app, as distinct from
/// <see cref="SessionOptions"/> which describes one particular chat session.
/// A model folder, a tools folder, and a colour theme are things you set
/// once and expect to stick around; the specific model and sampling
/// parameters for *this* session are not.
/// </summary>
public sealed class AppSettings
{
    /// <summary>Where the Model Browser starts looking by default.</summary>
    public string? DefaultModelFolder { get; set; }

    /// <summary>Where Options' "Tool DLLs folder" picker starts, and where ToolAssemblyLoader looks for ambient tool DLLs (see Options screen for the per-session explicit-path list, which is separate from this).</summary>
    public string? ToolsFolder { get; set; }

    /// <summary>Which built-in theme to render with. See <see cref="Theme"/>.</summary>
    public ThemeKind Theme { get; set; } = ThemeKind.HighContrastDark;

    [JsonIgnore]
    public static string SettingsFilePath
    {
        get
        {
            string configRoot = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            // ApplicationData can come back empty on some minimal Linux containers
            // with no XDG config dir set up at all; fall back to the user's home
            // directory rather than failing to ever persist anything.
            if (string.IsNullOrEmpty(configRoot))
                configRoot = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(configRoot, "SharpMind", "cui-settings.json");
        }
    }

    public static AppSettings Load()
    {
        try
        {
            var path = SettingsFilePath;
            if (!File.Exists(path)) return new AppSettings();
            var json = File.ReadAllText(path);
            return JsonSerializer.Deserialize<AppSettings>(json) ?? new AppSettings();
        }
        catch
        {
            // A corrupt or unreadable settings file shouldn't prevent the app
            // from starting — fall back to defaults rather than crash on launch.
            return new AppSettings();
        }
    }

    public bool Save(out string? error)
    {
        error = null;
        try
        {
            var path = SettingsFilePath;
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var json = JsonSerializer.Serialize(this, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, json);
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }
}
