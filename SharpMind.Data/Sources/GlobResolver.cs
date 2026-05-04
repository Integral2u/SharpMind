namespace SharpMind.Data.Sources;

/// <summary>
/// Resolves file paths from literal paths or glob patterns.
/// Supports * (single directory) and ** (recursive) wildcards.
/// Results are always returned in lexicographic order for reproducibility.
/// </summary>
public static class GlobResolver
{
    public static string[] Resolve(string pattern)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pattern);

        if (!pattern.Contains('*') && !pattern.Contains('?'))
            return File.Exists(pattern) ? [Path.GetFullPath(pattern)] : [];

        string root = GetRoot(pattern);
        string glob = pattern[root.Length..].TrimStart(
                                 Path.DirectorySeparatorChar,
                                 Path.AltDirectorySeparatorChar);
        bool recurse = glob.StartsWith("**");
        string filePattern = recurse
            ? glob[(glob.IndexOf(Path.DirectorySeparatorChar) + 1)..]
            : glob;

        var option = recurse ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

        return Directory.Exists(root)
            ? [.. Directory.GetFiles(root, filePattern, option).Order()]
            : [];
    }

    public static string[] ResolveMany(IEnumerable<string> paths)
        => [.. paths.SelectMany(p => Resolve(p)).Distinct().Order()];

    private static string GetRoot(string pattern)
    {
        int wildcard = pattern.IndexOfAny(['*', '?']);
        string dir = Path.GetDirectoryName(pattern[..wildcard]) ?? ".";
        return Path.GetFullPath(dir);
    }
}
