namespace SharpMind.CUI.App;

/// <summary>
/// Central resolution of SharpMind's default user folders. Instead of scattering
/// paths built on <c>%APPDATA%</c> or <c>%USERPROFILE%</c> across settings and
/// job documents, everything that ships with a default location derives from one
/// tree under the user's Documents folder:
///
/// <list type="bullet">
///   <item><c>Documents\SharpMind</c> — root.</item>
///   <item><c>Documents\SharpMind\Training</c> — saved training jobs (*.smmt),
///   tokenizer caches, and derived training folders.</item>
///   <item><c>Documents\SharpMind\Chat Sessions</c> — saved chat sessions and
///   option presets (SessionOptions as *.json).</item>
///   <item><c>Documents\SharpMind\Models</c> — where the Model Browser starts and
///   where exported training models land by default.</item>
/// </list>
///
/// All of these are <em>defaults</em>; the moment the user records a real path
/// (a custom export folder, a chosen model folder) it takes precedence, so the
/// folders can drift anywhere the user wants over time.
/// </summary>
public static class SharpMindPaths
{
    /// <summary>Tests may inject a replacement root before any path is read.</summary>
    public static string? OverrideRoot { get; set; }

    private static string GetBase()
    {
        if (!string.IsNullOrWhiteSpace(OverrideRoot)) return OverrideRoot;
        string documents = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        return string.IsNullOrEmpty(documents)
            ? Environment.GetFolderPath(Environment.SpecialFolder.UserProfile)
            : documents;
    }

    public static string Root => Path.Combine(GetBase(), "SharpMind");
    public static string Training => Path.Combine(Root, "Training");
    public static string ChatSessions => Path.Combine(Root, "Chat Sessions");
    public static string Models => Path.Combine(Root, "Models");

    /// <summary>Creates the three default subfolders; safe to call at startup.</summary>
    public static void EnsureCreated()
    {
        Directory.CreateDirectory(Root);
        Directory.CreateDirectory(Training);
        Directory.CreateDirectory(ChatSessions);
        Directory.CreateDirectory(Models);
    }
}