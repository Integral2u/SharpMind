using SharpMind.Data.Sources.PseudoLanguage;

namespace SharpMind.Tests.Data;

public sealed class PseudoLanguageSourceTests
{
    [Fact]
    public async Task ReadAsync_YieldsNonEmptyDocuments()
    {
        await using var source = new PseudoLanguageSource(
            vocabSize: 1_000,
            rootMorphemes: 100,
            affixes: 10,
            sequenceCount: 50,
            level: ComplexityLevel.Syntactic);

        var docs = new List<string>();
        await foreach (var text in source.ReadAsync())
            docs.Add(text);

        Assert.NotEmpty(docs);
        Assert.All(docs, d => Assert.False(string.IsNullOrWhiteSpace(d)));
        Assert.True(docs.Count <= 50);
    }

    [Fact]
    public void EstimatedCount_ReflectsRequestedCount()
    {
        var source = new PseudoLanguageSource(
            vocabSize: 1_000,
            rootMorphemes: 100,
            affixes: 10,
            sequenceCount: 123);

        Assert.Equal(123, source.EstimatedCount);
        source.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }

    [Fact]
    public void Description_IncludesLevelAndVocabulary()
    {
        var source = new PseudoLanguageSource(
            vocabSize: 2_000,
            rootMorphemes: 200,
            affixes: 15,
            sequenceCount: 10,
            level: ComplexityLevel.Patterns);

        Assert.Contains("Patterns", source.Description);
        Assert.Contains("vocab=", source.Description);
        source.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-5)]
    public void Ctor_RejectsInvalidVocabSize(int vocabSize)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PseudoLanguageSource(vocabSize, 100, 10, 10));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void Ctor_RejectsInvalidSequenceCount(int sequenceCount)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new PseudoLanguageSource(1_000, 100, 10, sequenceCount));
    }

    [Fact]
    public void Ctor_DefaultsMatchProductionConfig()
    {
        var source = new PseudoLanguageSource(
            vocabSize: 5_000,
            rootMorphemes: 300,
            affixes: 20,
            sequenceCount: 10_000);

        Assert.Contains("Syntactic", source.Description);
        Assert.Contains("n=10000", source.Description);
        source.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }
}