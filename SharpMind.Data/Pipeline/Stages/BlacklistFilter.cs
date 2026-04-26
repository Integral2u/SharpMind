using System.Text.RegularExpressions;

namespace SharpMind.Data.Pipeline.Stages;

public sealed class BlacklistFilter : ICleaningStage
{
    private readonly Regex _pattern;

    /// <param name="terms">Words or phrases to blacklist.</param>
    /// <param name="wholeWord">
    /// When true (default), only matches whole words — "ass" will not match "assignment".
    /// When false, matches anywhere in the document.
    /// </param>
    public BlacklistFilter(IEnumerable<string> terms, bool wholeWord = true, bool caseSensitive = false)
    {
        ArgumentNullException.ThrowIfNull(terms);
        var escaped = terms.Select(Regex.Escape);
        string pattern = wholeWord
            ? $@"\b(?:{string.Join('|', escaped)})\b"
            : $@"(?:{string.Join('|', escaped)})";

        _pattern = new Regex(pattern,
            caseSensitive? RegexOptions.Compiled : RegexOptions.Compiled | RegexOptions.IgnoreCase,
            TimeSpan.FromSeconds(2));
    }

    /// <param name="filePath">Path to a plain text file with one term per line.</param>
    /// <param name="wholeWord">See <see cref="BlacklistFilter(IEnumerable{string}, bool)"/>.</param>
    public BlacklistFilter(string filePath, bool wholeWord = true)
        : this(File.ReadLines(filePath)
                   .Select(l => l.Trim())
                   .Where(l => l.Length > 0 && !l.StartsWith('#')),
               wholeWord)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);
    }

    public string Name => "BlacklistFilter";
    public string? Process(string document) => _pattern.IsMatch(document) ? null : document;
}

