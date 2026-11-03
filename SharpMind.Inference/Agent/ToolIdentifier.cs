using System.Text.Json.Nodes;

namespace SharpMind.Inference.Agent;

/// <summary>
/// Identifier matching shared by tool-name resolution and argument binding.
/// Normalization removes separators and case so the spellings a model mixes up
/// (<c>action_type</c>, <c>actionType</c>, <c>Action Type</c>) compare equal.
/// Fuzzy matching is opt-in: only <see cref="Chat.ToolCallRecoveryMode.Loose"/>
/// uses it, and only when a unique candidate is within a small edit distance.
/// </summary>
internal static class ToolIdentifier
{
    /// <summary>Case- and separator-insensitive form used for identifier equality.</summary>
    public static string Normalize(string value)
        => value.Replace("_", string.Empty)
                .Replace("-", string.Empty)
                .Replace(" ", string.Empty)
                .ToLowerInvariant();

    /// <summary>
    /// Optimal-string-alignment (Damerau-Levenshtein) distance: insertions,
    /// deletions, substitutions and adjacent transpositions each cost one.
    /// Long inputs are refused rather than measured; this only runs on tool and
    /// argument names.
    /// </summary>
    public static int Distance(string a, string b)
    {
        if (string.Equals(a, b, StringComparison.Ordinal)) return 0;
        if (a.Length == 0) return b.Length;
        if (b.Length == 0) return a.Length;
        if (a.Length > 128 || b.Length > 128) return int.MaxValue;

        var d = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i <= a.Length; i++) d[i, 0] = i;
        for (int j = 0; j <= b.Length; j++) d[0, j] = j;

        for (int i = 1; i <= a.Length; i++)
        {
            for (int j = 1; j <= b.Length; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                int value = Math.Min(
                    Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1),
                    d[i - 1, j - 1] + cost);

                if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1])
                    value = Math.Min(value, d[i - 2, j - 2] + 1);

                d[i, j] = value;
            }
        }

        return d[a.Length, b.Length];
    }

    /// <summary>
    /// True when a distance is small enough to be a model typo rather than a
    /// different name: at most one edit for short names, two for longer ones,
    /// and never more than a quarter of the shorter name.
    /// </summary>
    public static bool IsWithinFuzzyThreshold(string a, string b, int distance)
    {
        int length = Math.Min(a.Length, b.Length);
        if (length < 4 || distance == int.MaxValue) return false;
        int max = length <= 8 ? 1 : 2;
        if (distance > max) return false;
        return distance <= length * 0.25;
    }

    /// <summary>Convenience overload for comparing one normalized name with itself.</summary>
    public static bool IsWithinFuzzyThreshold(string normalized, int distance)
        => IsWithinFuzzyThreshold(normalized, normalized, distance);

    /// <summary>
    /// True when a value names a filesystem location. Used as a safety guard in
    /// Loose recovery: a guessed tool call must never be pointed at a path. Bare
    /// filenames with an extension count; URLs do not.
    /// </summary>
    public static bool LooksLikePath(string value)
    {
        if (HasPathStructure(value)) return true;

        var s = value.Trim();
        if (s.Length == 0) return false;

        // A bare filename such as "notes.txt". Deliberately conservative about
        // the extension: letters only, one to five characters.
        if (!s.Contains(' ') && !s.Contains("://", StringComparison.Ordinal))
        {
            int dot = s.LastIndexOf('.');
            if (dot > 0 && dot < s.Length - 1)
            {
                var extension = s[(dot + 1)..];
                if (extension.Length is >= 1 and <= 5 && extension.All(char.IsLetter))
                    return true;
            }
        }

        return false;
    }

    /// <summary>
    /// True when a value has unmistakable path structure: separators, a drive
    /// letter, a UNC prefix, or an explicit relative/dot prefix. Unlike
    /// <see cref="LooksLikePath"/> this does not flag a bare filename, so free
    /// text such as a choice detail mentioning "notes.txt" is not mistaken for
    /// a filesystem target.
    /// </summary>
    public static bool HasPathStructure(string value)
    {
        var s = value.Trim();
        if (s.Length == 0) return false;
        if (s.Contains('\\')) return true;
        if (s is "." or "..") return true;
        if (s.StartsWith('/') || s.StartsWith("./") || s.StartsWith("../") || s.StartsWith("~/"))
            return true;
        if (s.Length >= 2 && char.IsLetter(s[0]) && s[1] == ':') return true;
        return s.Contains('/') && !s.Contains("://", StringComparison.Ordinal);
    }

    /// <summary>True when an argument name marks a filesystem location.</summary>
    public static bool IsPathLikeKey(string key)
    {
        var normalized = Normalize(key);
        return normalized is "path" or "file" or "filepath" or "filename"
            or "directory" or "dir" or "folder";
    }

    /// <summary>
    /// True when a call argument targets a filesystem location: any value with
    /// path structure, or a bare filename under a path-named argument.
    /// </summary>
    public static bool ContainsPathLikeArgument(JsonObject args)
    {
        foreach (var kv in args)
        {
            if (kv.Value is not JsonValue value
                || !value.TryGetValue<string>(out var text))
            {
                continue;
            }

            if (HasPathStructure(text)) return true;
            if (IsPathLikeKey(kv.Key) && LooksLikePath(text)) return true;
        }
        return false;
    }
}
