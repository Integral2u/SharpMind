using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Training;

/// <summary>
/// The vocabulary suggester analyses a corpus and recommends the smallest BPE
/// vocabulary that actually compresses it — reporting "no growth needed" once
/// adding merges stops paying for itself.
/// </summary>
public sealed class VocabSuggesterTests
{
    [Fact]
    public async Task Suggest_RepetitiveCorpus_GrowsAboveBase()
    {
        // A corpus with heavy word repetition: merges are worthwhile, so the
        // suggestion should land above the base vocab and report growth.
        var docs = Enumerable.Range(0, 200)
            .Select(_ => "internationalization internationalization localization localization")
            .ToArray();

        var suggestion = await VocabSuggester.SuggestAsync(new InMemorySource(docs), currentVocabSize: 0);

        Assert.True(suggestion.GrowthWorthwhile);
        Assert.True(suggestion.SuggestedVocabSize > 0);
        // Adding the suggested tokens must not increase the corpus token count.
        Assert.True(suggestion.CorpusTokensSuggested <= suggestion.CorpusTokensNow);
        Assert.True(suggestion.TokensPerWordSuggested <= suggestion.TokensPerWordNow + 1e-9);
    }

    [Fact]
    public async Task Suggest_NoGainOverCurrent_MarksNotWorthwhile()
    {
        var docs = Enumerable.Range(0, 100)
            .Select(_ => "alpha beta gamma alpha beta gamma")
            .ToArray();

        // Ask for a suggestion, then re-ask treating the already-suggested size as
        // the current one: the second pass should find no further growth needed.
        var first = await VocabSuggester.SuggestAsync(new InMemorySource(docs), currentVocabSize: 0);
        var second = await VocabSuggester.SuggestAsync(new InMemorySource(docs), currentVocabSize: first.SuggestedVocabSize);

        Assert.False(second.GrowthWorthwhile);
    }

    [Fact]
    public async Task Suggest_SingleCharacterWords_DoesNotGrow()
    {
        // Words of length one have no byte pairs, so no merge can save anything.
        var docs = Enumerable.Range('A', 26).Select(c => ((char)c).ToString()).ToArray();
        var suggestion = await VocabSuggester.SuggestAsync(new InMemorySource(docs), currentVocabSize: 0);

        Assert.False(suggestion.GrowthWorthwhile);
        Assert.Equal(0, suggestion.Analysis.MaxSavings);
    }
}
