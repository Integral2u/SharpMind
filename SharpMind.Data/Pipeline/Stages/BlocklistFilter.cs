using SharpMind.Data.Metadata;

namespace SharpMind.Data.Pipeline.Stages;

[ComponentKind("Blocklist Filter", "Drops documents containing any pattern from a word-list file.")]
public sealed class BlocklistFilter : ICleaningStage
{
    private readonly HashSet<string> _patterns;
    private readonly StringComparison _comparison;

    public BlocklistFilter(IEnumerable<string> patterns, bool caseSensitive = false)
    {
        ArgumentNullException.ThrowIfNull(patterns);
        var comparer = caseSensitive
            ? null
            : StringComparer.OrdinalIgnoreCase;
        _patterns = new HashSet<string>(patterns, comparer);
        _comparison = caseSensitive
            ? StringComparison.Ordinal
            : StringComparison.OrdinalIgnoreCase;
    }

    public BlocklistFilter(
        [FileChooser("*.txt", "Word-list file; one pattern per line, # comments allowed")] string filePath,
        [DefaultValue("false")] bool caseSensitive = false)
        : this(File.ReadLines(filePath)
                   .Select(l => l.Trim())
                   .Where(l => l.Length > 0 && !l.StartsWith('#')),
               caseSensitive)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
    }
    public string Name => $"BlocklistFilter(count={_patterns.Count})";

    public string? Process(string document)
    {
        if (document is null) return null;
        foreach (var pattern in _patterns)
        {
            if (document.Contains(pattern, _comparison))
                return null;
        }
        return document;
    }
}