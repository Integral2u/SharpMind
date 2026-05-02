namespace SharpMind.Data.Sources.PseudoLanguage;

public sealed record VocabConfig
{
    public int VocabSize { get; init; } = 5_000;
    public int RootMorphemes { get; init; } = 300;
    public int Affixes { get; init; } = 20;
    public int MaxWordLength { get; init; } = 12;
    public int MinWordLength { get; init; } = 3;

    public static VocabConfig Tiny => new() { VocabSize = 500, RootMorphemes = 50, Affixes = 10 };
    public static VocabConfig Small => new() { VocabSize = 2_000, RootMorphemes = 150, Affixes = 15 };
    public static VocabConfig Medium => new() { VocabSize = 5_000, RootMorphemes = 300, Affixes = 20 };
    public static VocabConfig Large => new() { VocabSize = 20_000, RootMorphemes = 800, Affixes = 30 };
    public static VocabConfig Huge => new() { VocabSize = 100_000, RootMorphemes = 2_000, Affixes = 50 };
}
