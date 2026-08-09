namespace SharpMind.Data.Metadata;

/// <summary>
/// Marks a constructor parameter as a file path. The training wizard presents a
/// file picker configured with <see cref="Pattern"/> (e.g. <c>"*.csv"</c>) and
/// fills the parameter with the chosen path.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
public sealed class FileChooserAttribute(string pattern = "*.*", string? help = null) : Attribute
{
    /// <summary>File-picker glob pattern, e.g. <c>"*.json"</c> or <c>"*.txt"</c>.</summary>
    public string Pattern { get; } = pattern;

    /// <summary>Optional help text shown alongside the picker.</summary>
    public string? Help { get; } = help;
}

/// <summary>
/// Marks a constructor parameter as a directory path. The training UI shows a
/// folder picker and fills the parameter with the chosen directory.
/// </summary>
[AttributeUsage(AttributeTargets.Parameter, Inherited = false)]
public sealed class FolderChooserAttribute(string? help = null) : Attribute
{
    /// <summary>Optional help text shown alongside the picker.</summary>
    public string? Help { get; } = help;
}