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

        int wildcard = pattern.IndexOfAny(['*', '?']);
        string dir = wildcard > 0 ? Path.GetDirectoryName(pattern[..wildcard]) ?? "" : "";
        string root = Path.GetFullPath(dir.Length == 0 ? "." : dir);
        string glob = (dir.Length == 0
                ? pattern
                : pattern[dir.Length..]).TrimStart('/', '\\');
        bool recurse = glob.StartsWith("**");
        string filePattern = recurse
            ? StripRecursionPrefix(glob)
            : glob;

        var option = recurse ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;

        return Directory.Exists(root)
            ? [.. Directory.GetFiles(root, filePattern, option).Order()]
            : [];
    }

    public static string[] ResolveMany(IEnumerable<string> paths)
        => [.. paths.SelectMany(p => Resolve(p)).Distinct().Order()];

    // "**" may be followed by either a backslash or a forward slash, so split
    // with an OS-agnostic separator rather than Path.DirectorySeparatorChar.
    private static string StripRecursionPrefix(string glob)
    {
        int sep = glob.IndexOfAny(['/', '\\']);
        return sep >= 0 ? glob[(sep + 1)..] : "*";
    }
}
