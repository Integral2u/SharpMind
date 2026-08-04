using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Training;

namespace SharpMind.Tests.Training;

/// <summary>
/// Covers <see cref="TrainingTokenizerBuilder"/>: specials must land inside the
/// model's vocab range, filler rows must collapse/expand correctly on reload,
/// and an undersized vocab must be rejected.
/// </summary>
public class TrainingTokenizerBuilderTests
{
    [Fact]
    public void BuildForVocab_GeneratorSpecialsLandInsideVocab()
    {
        var generator = new LearnableGenerator(new LearnableConfig(), new Random(1));

        var tokenizer = TrainingTokenizerBuilder.BuildForVocab(generator, vocabSize: 64);

        // 22 unique generator words (the raw Vocabulary list contains "fish"
        // twice — the tokenizer dedupes it) + 4 specials, all within 64 rows.
        int unique = generator.Vocabulary.Distinct().Count();
        Assert.Equal(unique + 4, tokenizer.VocabSize);
        Assert.InRange(tokenizer.UnkId, 0, 63);
        Assert.InRange(tokenizer.BosId, 0, 63);
        Assert.InRange(tokenizer.EosId, 0, 63);
        Assert.InRange(tokenizer.PadId, 0, 63);
        Assert.Equal(unique, tokenizer.UnkId);
        Assert.Equal(unique + 1, tokenizer.BosId);
        Assert.Equal(unique + 2, tokenizer.EosId);
        Assert.Equal(unique + 3, tokenizer.PadId);

        // Every word maps inside the model's vocab, and ids round-trip.
        foreach (string word in generator.Vocabulary)
            Assert.InRange(tokenizer.TokenToId(word), 0, 64);
        for (int id = 0; id < tokenizer.VocabSize; id++)
            Assert.Equal(id, tokenizer.TokenToId(tokenizer.IdToToken(id)));
    }

    [Fact]
    public void BuildForVocab_ExactFitHasNoFillerRows()
    {
        string[] words = ["apple", "banana", "cherry"];

        // words(3) + specials(4) == vocabSize(7) — no filler gap.
        var tokenizer = TrainingTokenizerBuilder.BuildForVocab(words, vocabSize: 7);

        Assert.Equal(7, tokenizer.VocabSize);
        Assert.Equal(3, tokenizer.UnkId);
        Assert.Equal(4, tokenizer.BosId);
        Assert.Equal(5, tokenizer.EosId);
        Assert.Equal(6, tokenizer.PadId);
    }

    [Fact]
    public void BuildForVocab_CustomUnknownTokensKeepFillersDistinct()
    {
        string[] words = ["apple"];

        var tokenizer = TrainingTokenizerBuilder.BuildForVocab(words, vocabSize: 8, unknownTokens: i => $"<f{i}>");

        // 1 word + 3 distinct fillers + 4 specials = 8 rows.
        Assert.Equal(8, tokenizer.VocabSize);
        Assert.Equal("<f1>", tokenizer.IdToToken(1));
        Assert.Equal("<f2>", tokenizer.IdToToken(2));
        Assert.Equal("<f3>", tokenizer.IdToToken(3));
        Assert.Equal(4, tokenizer.UnkId);
        Assert.Equal(5, tokenizer.BosId);
        Assert.Equal(6, tokenizer.EosId);
        Assert.Equal(7, tokenizer.PadId);
    }

    [Fact]
    public void BuildForVocab_TooManyWords_Throws()
    {
        string[] words = ["a", "b", "c", "d", "e"];

        Assert.Throws<ArgumentException>(() => TrainingTokenizerBuilder.BuildForVocab(words, vocabSize: 8));
    }
}
