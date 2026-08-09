using System.Text.RegularExpressions;
using SharpMind.Data.Metadata;

namespace SharpMind.Data.Pipeline.Stages;
/// <summary>Keeps only documents that match a regex pattern.</summary>
[ComponentKind("Regex Keep Filter", "Keeps only documents matching a regex.")]
public sealed class RegexKeepFilter(
    [Tooltip("Regex pattern; matching documents are kept.")] string pattern,
    [DefaultValue("None")] RegexOptions options = RegexOptions.None) : ICleaningStage
{
    private readonly Regex _pattern = new(pattern,
               options | RegexOptions.Compiled, TimeSpan.FromSeconds(2));

    public string Name => $"RegexKeep({_pattern})";
    public string? Process(string document)
        => _pattern.IsMatch(document) ? document : null;
}