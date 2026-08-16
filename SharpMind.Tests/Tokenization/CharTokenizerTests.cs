using SharpMind.Data.Sources;
using SharpMind.Tokenization;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// Covers the character-level tokenizer built by
/// <see cref="TokenizationPipeline.TrainCharacterAsync"/>: the vocabulary is
/// the sorted set of distinct corpus characters (each character is one token),
/// BOS/EOS are supported, unknown characters map to UNK, and the char mode
/// round-trips through the native JSON format.
/// </summary>
public sealed class CharTokenizerTests
{
    private static Tokenizer Train(string corpus)
    {
        using var dir = new TempDirectory();
        string path = dir.Write("corpus.txt", corpus);
        var source = new TextFileSource(path, TextFileSource.DocumentMode.FilePerDoc);
        return TokenizationPipeline.TrainCharacterAsync(source).GetAwaiter().GetResult();
    }

    [Fact]
    public void Training_VocabIsSortedUniqueCharsPlusSpecials()
    {
        // unique chars of "abca b\n": '\n', ' ', 'a', 'b', 'c' → 5 chars + 4 specials
        var tokenizer = Train("abca b\n");
        Assert.Equal(9, tokenizer.VocabSize);

        // characters come first, sorted by code point; specials live at the tail
        Assert.Equal("\n", tokenizer.IdToToken(0));
        Assert.Equal(" ", tokenizer.IdToToken(1));
        Assert.Equal("a", tokenizer.IdToToken(2));
        Assert.Equal("b", tokenizer.IdToToken(3));
        Assert.Equal("c", tokenizer.IdToToken(4));
        Assert.True(tokenizer.BosId >= 5);
        Assert.True(tokenizer.EosId >= 5);
        Assert.True(tokenizer.PadId >= 5);
    }

    [Fact]
    public void Encode_OneIdPerCharacter_AndDecodeRoundtrips()
    {
        // corpus contains 'é' so a non-ASCII character is part of the vocabulary
        var tokenizer = Train("héllo code");

        var plain = tokenizer.Encode("hello");
        Assert.Equal(5, plain.Length);
        Assert.Equal("hello", tokenizer.Decode(plain));

        var framed = tokenizer.Encode("olle h", addBos: true, addEos: true);
        Assert.Equal(tokenizer.BosId, framed[0]);
        Assert.Equal(tokenizer.EosId, framed[^1]);
        Assert.Equal(8, framed.Length);
        Assert.Equal("olle h", tokenizer.Decode(framed));
    }

    [Fact]
    public void Encode_UnknownCharacter_FallsBackToUnk()
    {
        var tokenizer = Train("abc");
        var ids = tokenizer.Encode("abz");
        Assert.Equal(tokenizer.TokenToId("a"), ids[0]);
        Assert.Equal(tokenizer.TokenToId("b"), ids[1]);
        Assert.Equal(tokenizer.UnkId, ids[2]);
    }

    [Fact]
    public void Json_RoundTrips_PreservesCharMode()
    {
        // corpus includes a space so "a b c" round-trips as five real characters
        var tokenizer = Train("a b c");

        var reloaded = Tokenizer.FromJson(tokenizer.ToJson());

        Assert.Equal(tokenizer.VocabSize, reloaded.VocabSize);
        Assert.Equal(tokenizer.TokenToId("a"), reloaded.TokenToId("a"));

        // char mode survives the round trip: one ID per character, no merges
        Assert.Equal(2, reloaded.Encode("ab").Length);
        Assert.Equal(5, reloaded.Encode("a b c").Length);
        Assert.Equal("a b c", reloaded.Decode(reloaded.Encode("a b c")));
    }
}