using SharpMind.Data.Sources;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Serialisation;
using SharpMind.Tokenization.Vocab;
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

    [Fact]
    public async Task Grow_SegmentsWithTheGrownTokenizersPreTokeniser_NotTheTrainers()
    {
        // The whitespace-split tokenizer sees "x-y" as one word; the default trainer's GPT-2 split sees
        // "x", "-", "y" and " x". The merges must come from the segmentation the grown tokenizer will
        // produce at encode time: "-y" then "x-y", never "Ġx".
        var original = await new BpeTrainer(targetVocabSize: 262, minFrequency: 1, preTokeniser: new WhitespacePreTokeniser())
            .TrainAsync(Docs("the quick brown fox"));
        int target = original.Vocab.Size + 2;

        var grown = await new BpeTrainer(target).GrowAsync(original, Docs("x-y x-y x-y x-y"), target);

        var learned = grown.Vocab.AllTokens.Skip(original.Vocab.Size).ToArray();
        Assert.Equal(["-y", "x-y"], learned);
        Assert.DoesNotContain(Vocabulary.ByteTokenString(0x20) + "x", learned);
        Assert.Same(original.PreTokeniser, grown.PreTokeniser);
    }

    [Fact]
    public async Task Grow_KeepsTheGgufNames()
    {
        var original = await Trainer(280).TrainAsync(Docs("the quick brown fox"));
        original.GgufPreTokenizer = "qwen2";
        original.GgufTokenizerModel = "gpt2";

        var grown = await Trainer(300).GrowAsync(original, Docs("alpha beta gamma alpha beta"), 300);

        Assert.Equal("qwen2", grown.GgufPreTokenizer);
        Assert.Equal("gpt2", grown.GgufTokenizerModel);
        Assert.False(grown.Encoder.UseSentencePieceMerge);
        Assert.Equal("qwen2", TokenizerFile.FromJson(TokenizerFile.ToJson(grown)).GgufPreTokenizer);
    }

    [Fact]
    public async Task Grow_ByteLevelOriginalWithoutMerges_EncodesWithTheNewMerges()
    {
        // 260 = specials + 256 byte tokens: nothing learned, which the encoder's heuristic reads as
        // SentencePiece. Once grown the model has merges and must use them.
        var original = await new BpeTrainer(targetVocabSize: 260).TrainAsync(Docs("a"));
        Assert.Empty(original.Merges);

        var grown = await new BpeTrainer(261).GrowAsync(original, Docs("ab ab"), 261);

        Assert.True(grown.Vocab.Contains("ab"));
        Assert.Equal([grown.Vocab.GetId("ab")], new Tokenizer(grown).Encode("ab", addBos: false, addEos: false));
    }

    [Theory]
    [InlineData(true, "llama")]   // a SentencePiece GGUF: scores and the model name
    [InlineData(false, "llama")]  // the model name alone
    [InlineData(false, null)]     // neither: the standalone ▁ token decides
    public async Task Grow_Throws_OnSentencePieceTokenizer(bool withScores, string? ggufModel)
    {
        // Score-ranked merges cannot be continued by appending rank-ordered ones.
        var sp = TokenizerFile.FromJson(Tokenizer.FromGguf(
            tokens: ["<unk>", "<s>", "</s>", "\u2581", "a", "b", "\u2581a"], merges: null, tokenTypes: null,
            bosId: 1, eosId: 2, scores: withScores ? [0f, 0f, 0f, 0f, 0f, 0f, -1f] : null, tokenizerModel: ggufModel).ToJson());
        Assert.True(sp.Encoder.UseSentencePieceMerge);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new BpeTrainer(300).GrowAsync(sp, Docs("ab ab"), 300));
        Assert.Contains("SentencePiece", ex.Message);
    }
}
