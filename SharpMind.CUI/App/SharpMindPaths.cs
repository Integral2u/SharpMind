namespace SharpMind.CUI.App;

/// <summary>
/// Central resolution of SharpMind's default user folders. Instead of scattering
/// paths built on <c>%APPDATA%</c> or <c>%USERPROFILE%</c> across settings and
/// job documents, everything that ships with a default location derives from one
/// tree at the root of the user's home directory:
///
/// <list type="bullet">
///   <item><c>%USERPROFILE%\SharpMind</c> — root.</item>
///   <item><c>%USERPROFILE%\SharpMind\Training</c> — saved training jobs (*.smmt),
///   tokenizer caches, and derived training folders.</item>
///   <item><c>%USERPROFILE%\SharpMind\Chat Sessions</c> — saved chat sessions and
///   option presets (SessionOptions as *.json).</item>
///   <item><c>%USERPROFILE%\SharpMind\Models</c> — where the Model Browser starts
///   and where exported training models land by default.</item>
///   <item><c>%USERPROFILE%\SharpMind\Corpus</c> — default browse start for
///   training data: drop corpora (*.txt, *.jsonl, …) here to use them.</item>
/// </list>
///
/// The root sits beside — never inside — the user's Documents folder. This
/// deliberately avoids two Windows gotchas: Controlled Folder Access (ransomware
/// protection) blocks unrecognized apps from writing to Documents/Desktop/etc.,
/// and the OneDrive Known Folder Move would otherwise sweep multi-GB checkpoints
/// into cloud sync. Barely under %USERPROFILE% it is neither a protected folder
/// nor a synced one.
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
        string profile = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return string.IsNullOrEmpty(profile)
            ? Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments)
            : profile;
    }

    public static string Root => Path.Combine(GetBase(), "SharpMind");
    public static string Training => Path.Combine(Root, "Training");
    public static string ChatSessions => Path.Combine(Root, "Chat Sessions");
    public static string Models => Path.Combine(Root, "Models");

    /// <summary>Default landing spot for training corpora (*.txt, *.jsonl, …).</summary>
    public static string Corpus => Path.Combine(Root, "Corpus");

    /// <summary>Creates the default subfolders; safe to call at startup.</summary>
    public static void EnsureCreated()
    {
        Directory.CreateDirectory(Root);
        Directory.CreateDirectory(Training);
        Directory.CreateDirectory(ChatSessions);
        Directory.CreateDirectory(Models);
        Directory.CreateDirectory(Corpus);
    }
}