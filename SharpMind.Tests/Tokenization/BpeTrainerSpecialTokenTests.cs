using System.Runtime.CompilerServices;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Vocab;
using Xunit;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// The BPE trainer accepts a reserved special-token list: reserved tokens take
/// the lowest IDs (before any byte tokens or learned merges) and are matched
/// verbatim rather than being shredded by BPE.
/// </summary>
public sealed class BpeTrainerSpecialTokenTests
{
    private static async IAsyncEnumerable<string> Docs(params string[] documents)
    {
        foreach (string document in documents)
        {
            await Task.Yield();
            yield return document;
        }
    }

    [Fact]
    public async Task ReservedSpecials_GetLowestIds_AndAreMatchedVerbatim()
    {
        var specials = new SpecialTokens(new SpecialTokensConfig
        {
            Additional = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"],
        });

        var trainer = new BpeTrainer(
            targetVocabSize: 280,
            minFrequency: 1,
            preTokeniser: new WhitespacePreTokeniser(),
            specials: specials,
            byteLevel: false);

        var model = await trainer.TrainAsync(Docs(
            "hello world hello world <|im_start|> hi <|im_end|>",
            "the quick brown fox <|endoftext|> hello world hello"));

        var tokenizer = new Tokenizer(model);

        // Unk, Bos, Eos, Pad, then the three additional tokens - all first.
        Assert.Equal(7, specials.All.Count);
        for (int i = 0; i < specials.All.Count; i++)
            Assert.Equal(specials.All[i], tokenizer.IdToToken(i));

        // A special token in the middle of text is a single token, not BPE input.
        int start = tokenizer.TokenToId("<|im_start|>");
        int[] ids = tokenizer.Encode("hello<|im_start|>world");
        Assert.Equal(1, ids.Count(id => id == start));
    }

    [Fact]
    public async Task WithoutReservedSpecials_TheTrainerStillTrains()
    {
        var trainer = new BpeTrainer(
            targetVocabSize: 280,
            minFrequency: 1,
            preTokeniser: new WhitespacePreTokeniser(),
            byteLevel: false);

        var model = await trainer.TrainAsync(Docs("hello world hello world"));
        var tokenizer = new Tokenizer(model);

        Assert.True(tokenizer.VocabSize > 4);
        Assert.Equal("[UNK]", tokenizer.IdToToken(0));
    }
}
