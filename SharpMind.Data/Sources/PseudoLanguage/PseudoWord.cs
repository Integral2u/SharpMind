namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed class PseudoWord
{
    public string Text { get; set; } = "";
    public int TokenId { get; set; }
    public MorphemeCategory BaseCategory { get; set; }
    public (string BaseWord, MorphemeCategory Category)[] WordFamily { get; set; } = [];

    public string Base => WordFamily.Length > 0 ? WordFamily[0].BaseWord : Text;
}