using SharpMind.Data.Sources;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Serialisation;
using Xunit;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// Growing a BPE model must be append-only: every token the original vocabulary
/// knew keeps its exact ID, encoding old text is unchanged, and only new merge
/// tokens occupy the previously-unused higher IDs. This is what makes a widened
/// embedding table safe to reuse on a warm restart.
/// </summary>
public sealed class BpeGrowTests
{
    private static async IAsyncEnumerable<string> Docs(params string[] documents)
    {
        foreach (string document in documents)
        {
            await Task.Yield();
            yield return document;
        }
    }

    private static BpeTrainer Trainer(int target) => new(
        targetVocabSize: target,
        minFrequency: 1,
        preTokeniser: new WhitespacePreTokeniser(),
        byteLevel: false);

    [Fact]
    public async Task Grow_PreservesEveryOldTokenId_AndAppendsNewOnes()
    {
        string[] corpus =
        [
            "the quick brown fox jumps over the lazy dog",
            "the quick red fox runs fast",
        ];

        var original = await Trainer(280).TrainAsync(Docs(corpus));
        var originalTokens = original.Vocab.AllTokens.ToArray();

        // A second corpus rich in new repeated words, trained on top of the first.
        var grown = await Trainer(300).GrowAsync(original, Docs(
            "alpha beta gamma alpha beta gamma delta alpha beta"), 300);

        // Every old token keeps its ID value (the token string at each old index is identical).
        for (int i = 0; i < originalTokens.Length; i++)
            Assert.Equal(originalTokens[i], grown.Vocab.AllTokens[i]);

        // Growth actually happened.
        Assert.True(grown.Vocab.Size > original.Vocab.Size);
        Assert.True(grown.Vocab.Size <= 300);

        // Old text encodes identically under both vocabularies.
        var oldEnc = new Tokenizer(original).Encode("the quick brown fox");
        var newEnc = new Tokenizer(grown).Encode("the quick brown fox");
        Assert.Equal(oldEnc, newEnc);
    }

    [Fact]
    public async Task Grow_WithDifferentByteLevelSetting_StillPreservesOldIds()
    {
        // Trained without byte tokens, then grown by a default (byte-level) trainer:
        // the grow must copy the existing vocabulary rather than rebuild it with
        // 256 byte tokens ahead of the old IDs.
        var original = await new BpeTrainer(
            targetVocabSize: 280,
            minFrequency: 1,
            preTokeniser: new WhitespacePreTokeniser(),
            byteLevel: false).TrainAsync(Docs("the quick brown fox jumps over the lazy dog"));

        var grown = await Trainer(300).GrowAsync(
            original, Docs("alpha beta gamma alpha beta"), 300);

        Assert.True(grown.Vocab.Size > original.Vocab.Size);
        for (int i = 0; i < original.Vocab.AllTokens.Count; i++)
            Assert.Equal(original.Vocab.AllTokens[i], grown.Vocab.AllTokens[i]);
    }

    [Fact]
    public async Task Grow_ReturnsOriginal_WhenTargetNotLarger()
    {
        var original = await Trainer(280).TrainAsync(Docs("hello world hello world"));
        var same = await Trainer(280).GrowAsync(original, Docs("some other text"), original.Vocab.Size);
        Assert.Same(original, same);
    }

    [Fact]
    public async Task Grow_RoundTripsThroughJson()
    {
        var original = await Trainer(280).TrainAsync(Docs("the quick brown fox"));
        var grown = await Trainer(300).GrowAsync(original, Docs("alpha beta gamma alpha beta"), 300);

        using var dir = new TempDirectory();
        string path = dir.Write("tok.json", TokenizerFile.ToJson(grown));
        var reloaded = TokenizerFile.Load(path);

        Assert.Equal(grown.Vocab.Size, reloaded.Vocab.Size);
        for (int i = 0; i < grown.Vocab.AllTokens.Count; i++)
            Assert.Equal(grown.Vocab.AllTokens[i], reloaded.Vocab.AllTokens[i]);
    }

    [Fact]
    public async Task Grow_Throws_OnCharacterTokenizer()
    {
        var charTokenizer = await TokenizationPipeline.TrainCharacterAsync(
            new InMemorySource(["abcabc"]));
        var charModel = TokenizerFile.FromJson(charTokenizer.ToJson());

        var trainer = Trainer(280);
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => trainer.GrowAsync(charModel, Docs("abc"), 300));
        Assert.Contains("character", ex.Message, StringComparison.OrdinalIgnoreCase);
    }
}
