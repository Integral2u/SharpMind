using System.Text.Json;

namespace SharpMind.CUI.App;

/// <summary>
/// A saved configuration. The same file format covers two different things:
///
/// - "Save Options As..." / "Open preset...": SessionOptions only, no
///   notion of a running session — purely a starting point for a future
///   Launch, including for code-only/scripted use outside this UI.
/// - "Save Session...": the same SessionOptions plus a display name, saved
///   from an actually-running session so it can be reopened later against
///   the same model file. Reopening still goes through the normal load path
///   (and benefits from ModelCache if another session for the same file is
///   already open) — this format doesn't attempt to serialize the loaded
///   Transformer itself, only what's needed to load it again.
/// </summary>
public sealed class SavedSession
{
    public string Name { get; set; } = "Untitled";
    public required SessionOptions Options { get; set; }
    public DateTime SavedAtUtc { get; set; } = DateTime.UtcNow;

    public static string DefaultFolder
    {
        get
        {
            string root = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
            if (string.IsNullOrEmpty(root)) root = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
            return Path.Combine(root, "SharpMind", "sessions");
        }
    }

    public static bool Save(SavedSession saved, string path, out string? error)
    {
        error = null;
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var json = JsonSerializer.Serialize(saved, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(path, json);
            return true;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return false;
        }
    }

    public static SavedSession? Load(string path, out string? error)
    {
        error = null;
        try
        {
            var json = File.ReadAllText(path);
            var result = JsonSerializer.Deserialize<SavedSession>(json);
            if (result is null) error = "File did not contain a valid saved session.";
            return result;
        }
        catch (Exception ex)
        {
            error = ex.Message;
            return null;
        }
    }

    /// <summary>Lists *.json files in a folder for the Open-preset/Load-session pickers, sorted newest first.</summary>
    public static List<string> ListSaved(string folder)
    {
        if (!Directory.Exists(folder)) return [];
        return Directory.GetFiles(folder, "*.json")
            .OrderByDescending(File.GetLastWriteTimeUtc)
            .ToList();
    }
}
