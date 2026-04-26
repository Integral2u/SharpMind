namespace SharpMind.Data.Pipeline.Stages;

public sealed class ProfanityFilter : ICleaningStage
{
    private readonly HashSet<string> _patterns;
    private readonly StringComparison _comparison;

    public ProfanityFilter(IEnumerable<string> patterns, bool caseSensitive = false)
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

    public string Name => $"ProfanityFilter(count={_patterns.Count})";

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