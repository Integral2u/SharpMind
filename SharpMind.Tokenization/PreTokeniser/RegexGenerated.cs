using System.Text.RegularExpressions;

namespace SharpMind.Tokenization.PreTokeniser;

public partial class RegexGenerated
{
    [GeneratedRegex(@"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.ExplicitCapture)]
    public static partial Regex Gpt2Pattern { get; }

    [GeneratedRegex(@"(?i:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", RegexOptions.ExplicitCapture)]
    public static partial Regex Cl100kPattern { get; }
}
