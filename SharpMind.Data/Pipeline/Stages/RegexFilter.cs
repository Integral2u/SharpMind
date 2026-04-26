using System.Text.RegularExpressions;

namespace SharpMind.Data.Pipeline.Stages;
/// <summary>
/// Discards documents matching a regex pattern (deny-list style).
/// Use <see cref="RegexKeepFilter"/> to keep only matching documents.
/// </summary>
public sealed class RegexFilter(string pattern, RegexOptions options = RegexOptions.None) : ICleaningStage
{
    private readonly Regex _pattern = new(pattern,
               options | RegexOptions.Compiled, TimeSpan.FromSeconds(2));

    public string Name => $"RegexFilter({_pattern})";
    public string? Process(string document)
        => _pattern.IsMatch(document) ? null : document;
}