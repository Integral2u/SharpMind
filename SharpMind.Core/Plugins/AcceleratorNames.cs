namespace SharpMind.Core.Plugins;

/// <summary>
/// Canonical-name handling for <see cref="IAcceleratorPlugin.Name"/>s. A plugin may be renamed
/// over time (the ILGPU plugin shipped as <c>cuda</c> before being renamed <c>ilgpu</c>); stored
/// training jobs (<c>.smmt</c>) and session presets keep whatever name was current when they were
/// saved, so the host resolves those old names rather than breaking the load. Both
/// <c>TrainingEngineResolver</c> and <c>InferenceEngineResolver</c> canonicalize the requested name
/// before matching against the loaded plugins, and the CUI keeps stored legacy values selectable
/// and rewrites them to the canonical name on save.
/// </summary>
public static class AcceleratorNames
{
    /// <summary>
    /// Maps a stored/requested accelerator name to today's canonical spelling (case-insensitive),
    /// or returns it unchanged when there is no alias. E.g. <c>"cuda" -&gt; "ilgpu"</c>. Pass the
    /// already-trimmed name.
    /// </summary>
    public static string Canonicalize(string name) =>
        string.Equals(name, "cuda", StringComparison.OrdinalIgnoreCase) ? "ilgpu" : name;

    /// <summary>
    /// True when <paramref name="name"/> refers to the plugin whose canonical name is
    /// <paramref name="canonical"/> — i.e. it is the canonical name or one of its legacy aliases
    /// (case-insensitive).
    /// </summary>
    public static bool Matches(string name, string canonical) =>
        string.Equals(Canonicalize(name), canonical, StringComparison.OrdinalIgnoreCase);
}
