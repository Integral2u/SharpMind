namespace SharpMind.Inference.Agent;

/// <summary>
/// Pure path helpers for the path-aware file permission rule: a resource that
/// lies inside one of the session's "approved roots" is assumed accessible
/// without prompting, while a resource that escapes to a parent directory
/// (outside every approved root) must go through the Ask permission flow.
/// </summary>
public static class PermissionPathPolicy
{
    /// <summary>
    /// Normalizes a path to its canonical absolute form, or returns <see langword="null"/>
    /// when the path is empty or cannot be resolved.
    /// </summary>
    public static string? TryResolveFullPath(string path)
    {
        if (string.IsNullOrWhiteSpace(path)) return null;
        try
        {
            return Path.GetFullPath(path);
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
        {
            return null;
        }
    }

    /// <summary>
    /// True when <paramref name="fullPath"/> equals or is a sub-directory of at least one
    /// of the given roots. Comparison is case-insensitive on Windows; trailing separators
    /// are trimmed so a root and one of its direct children never collide.
    /// </summary>
    public static bool IsUnderRoot(string fullPath, IEnumerable<string> roots)
    {
        string? normalized = TryResolveFullPath(fullPath);
        if (normalized is null) return false;
        normalized = normalized.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

        foreach (var root in roots)
        {
            string? r = TryResolveFullPath(root);
            if (r is null) continue;
            r = r.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);

            if (string.Equals(normalized, r, StringComparison.OrdinalIgnoreCase)) return true;
            if (normalized.StartsWith(r + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)) return true;
        }
        return false;
    }

    /// <summary>
    /// Whether a resource (absolute or relative path) lies inside at least one approved root.
    /// Relative resources are probed against each root: the resource counts as inside a root
    /// when combining the root with it stays within that root.
    /// </summary>
    public static bool IsResourceInsideRoots(string resource, IEnumerable<string> roots)
    {
        if (string.IsNullOrWhiteSpace(resource)) return false;

        List<string> rootsList;
        try
        {
            rootsList = roots.Where(r => !string.IsNullOrWhiteSpace(r)).ToList();
        }
        catch (Exception ex) when (ex is ArgumentException or NotSupportedException)
        {
            return false;
        }
        if (rootsList.Count == 0) return false;

        if (Path.IsPathRooted(resource))
            return IsUnderRoot(resource, rootsList);

        foreach (var root in rootsList)
        {
            if (IsUnderRoot(Path.Combine(root, resource), [root])) return true;
        }
        return false;
    }
}
