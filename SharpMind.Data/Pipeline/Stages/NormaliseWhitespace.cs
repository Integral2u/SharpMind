using System.Text.RegularExpressions;

namespace SharpMind.Data.Pipeline.Stages;

/// <summary>Collapses runs of whitespace to single spaces and trims ends.</summary>
public sealed class NormaliseWhitespace : ICleaningStage
{
    public string Name => "NormaliseWhitespace";

    private static readonly Regex WhitespaceRun =
        new(@"\s+", RegexOptions.Compiled, TimeSpan.FromSeconds(1));

    public string? Process(string document)
    {
        string result = WhitespaceRun.Replace(document.Trim(), " ");
        return result.Length == 0 ? null : result;
    }
}