using System.Text.RegularExpressions;

namespace SharpMind.Data.Pipeline.Stages;
/// <summary>Keeps only documents that match a regex pattern.</summary>
public sealed class RegexKeepFilter(string pattern, RegexOptions options = RegexOptions.None) : ICleaningStage
{
    private readonly Regex _pattern = new(pattern,
               options | RegexOptions.Compiled, TimeSpan.FromSeconds(2));

    public string Name => $"RegexKeep({_pattern})";
    public string? Process(string document)
        => _pattern.IsMatch(document) ? document : null;
}