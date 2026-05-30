namespace SharpMind.Model.Format;

public sealed class TokenizerInfo
{
    public string Type { get; set; } = "bpe";
    public string? VocabFile { get; set; }
    public Dictionary<string, string>? SpecialTokens { get; set; }
}