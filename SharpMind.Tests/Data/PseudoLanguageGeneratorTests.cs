using SharpMind.Data.Sources.PseudoLanguage;

namespace SharpMind.Tests.Data;

public sealed class PseudoLanguageGeneratorTests
{
    [Fact]
    public void VocabConfig_Tiny_HasExpectedDefaults()
    {
        var config = VocabConfig.Tiny;
        
        Assert.Equal(500, config.VocabSize);
        Assert.Equal(50, config.RootMorphemes);
        Assert.Equal(10, config.Affixes);
    }

    [Fact]
    public void VocabConfig_Medium_HasExpectedDefaults()
    {
        var config = VocabConfig.Medium;
        
        Assert.Equal(5_000, config.VocabSize);
        Assert.Equal(300, config.RootMorphemes);
        Assert.Equal(20, config.Affixes);
    }

    [Fact]
    public void Generator_CreatesVocabWithSpecifiedSize()
    {
        var config = new VocabConfig { VocabSize = 1000, RootMorphemes = 100, Affixes = 10 };
        var gen = new PseudoLanguageGenerator(config);
        
        Assert.True(gen.VocabSize >= 500);
        Assert.True(gen.VocabSize <= 1100);
    }

    [Fact]
    public void Generator_ContainsRootWords()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        
        Assert.Contains(gen.Vocabulary, w => w.Text == "walk");
        Assert.Contains(gen.Vocabulary, w => w.Text == "run");
    }

    [Fact]
    public void Generator_ContainsDerivedWords()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        
        Assert.Contains(gen.Vocabulary, w => w.Text.EndsWith("er"));
        Assert.Contains(gen.Vocabulary, w => w.Text.EndsWith("ed"));
    }

    [Fact]
    public void Generator_ContainsNegations()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Medium);
        
        Assert.Contains(gen.Vocabulary, w => w.Text.StartsWith("un"));
    }

    [Fact]
    public void Generator_GenerateOptions_ReturnsValidSequences()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        var sequences = gen.GenerateSyntactic(10, ComplexityLevel.Options).ToList();
        
        Assert.Equal(10, sequences.Count);
        foreach (var seq in sequences)
        {
            Assert.NotNull(seq.RawText);
            Assert.NotEmpty(seq.TokenIds);
            Assert.Single(seq.GroundTruthIds);
            Assert.Equal(ComplexityLevel.Options, seq.Complexity);
        }
    }

    [Fact]
    public void Generator_GeneratePatterns_ReturnsValidSequences()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        var sequences = gen.GenerateSyntactic(10, ComplexityLevel.Patterns).ToList();
        
        Assert.Equal(10, sequences.Count);
        foreach (var seq in sequences)
        {
            Assert.NotNull(seq.RawText);
            Assert.NotEmpty(seq.TokenIds);
            Assert.Single(seq.GroundTruthIds);
            Assert.Equal(ComplexityLevel.Patterns, seq.Complexity);
        }
    }

    [Fact]
    public void Generator_GenerateSyntactic_ReturnsValidSequences()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        var sequences = gen.GenerateSyntactic(10, ComplexityLevel.Syntactic).ToList();
        
        Assert.Equal(10, sequences.Count);
        foreach (var seq in sequences)
        {
            Assert.NotNull(seq.RawText);
            Assert.True(seq.TokenIds.Length >= 3);
            Assert.Single(seq.GroundTruthIds);
            Assert.Equal(ComplexityLevel.Syntactic, seq.Complexity);
        }
    }

    [Fact]
    public void Generator_IdToText_ReturnsCorrectWord()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        
        var text = gen.IdToText(0);
        Assert.NotNull(text);
        Assert.NotEqual($"<UNK:0>", text);
    }

    [Fact]
    public void Generator_TextToId_ReturnsCorrectId()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        
        var word = gen.Vocabulary.FirstOrDefault(w => w.Text == "walk");
        if (word != null)
        {
            var id = gen.TextToId("walk");
            Assert.Equal(word.TokenId, id);
        }
    }

    [Fact]
    public void Generator_TextToId_ReturnsNegativeForUnknown()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        
        var id = gen.TextToId("xyznonexistentword123");
        Assert.Equal(-1, id);
    }

    [Fact]
    public void ModelSizeRecommendation_ProvidesReasonableValues()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        var rec = gen.GetModelSizeRecommendation();
        
        Assert.Equal(gen.VocabSize, rec.VocabSize);
        Assert.True(rec.EmbeddingDim > 0);
        Assert.True(rec.HiddenDim > 0);
        Assert.True(rec.NumLayers > 0);
        Assert.True(rec.EstimatedParams > 0);
    }

    [Fact]
    public void ModelSizeRecommendation_ScalesWithVocabSize()
    {
        var tiny = new PseudoLanguageGenerator(VocabConfig.Tiny);
        var medium = new PseudoLanguageGenerator(VocabConfig.Medium);
        
        var recTiny = tiny.GetModelSizeRecommendation();
        var recMedium = medium.GetModelSizeRecommendation();
        
        Assert.True(recMedium.EstimatedParams > recTiny.EstimatedParams);
        Assert.True(recMedium.EmbeddingDim > recTiny.EmbeddingDim);
    }

    [Fact]
    public void PseudoWord_HasCorrectBase()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        
        var walker = gen.Vocabulary.FirstOrDefault(w => w.Text == "walker");
        if (walker != null)
        {
            Assert.Equal("walk", walker.Base);
        }
    }

    [Fact]
    public void PseudoWord_WordFamily_ContainsRelatedWords()
    {
        var gen = new PseudoLanguageGenerator(VocabConfig.Tiny);
        
        var walk = gen.Vocabulary.FirstOrDefault(w => w.Text == "walk");
        if (walk != null)
        {
            var family = gen.Vocabulary
                .Where(w => w.WordFamily.Any(f => f.BaseWord == "walk"))
                .ToList();
            
            Assert.True(family.Count >= 1);
        }
    }
}