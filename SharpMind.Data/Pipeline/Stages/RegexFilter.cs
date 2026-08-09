using System.Text.RegularExpressions;
using SharpMind.Data.Metadata;

namespace SharpMind.Data.Pipeline.Stages;
/// <summary>
/// Discards documents matching a regex pattern (deny-list style).
/// Use <see cref="RegexKeepFilter"/> to keep only matching documents.
/// </summary>
[ComponentKind("Regex Filter", "Drops documents matching a regex (deny-list).")]
public sealed class RegexFilter(
    [Tooltip("Regex pattern; matching documents are dropped.")] string pattern,
    [DefaultValue("None")] RegexOptions options = RegexOptions.None) : ICleaningStage
{
    private readonly Regex _pattern = new(pattern,
               options | RegexOptions.Compiled, TimeSpan.FromSeconds(2));

    public string Name => $"RegexFilter({_pattern})";
    public string? Process(string document)
        => _pattern.IsMatch(document) ? null : document;
}