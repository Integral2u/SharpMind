using SharpMind.Inference;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using Xunit;

namespace SharpMind.Tests.Inference;

/// <summary>
/// A character whose UTF-8 bytes span several tokens must be emitted whole, not
/// lost to U+FFFD stripping. Uses the byte-level test vocabulary: token id b is
/// byte b.
/// </summary>
public sealed class IncrementalDetokenizerTests
{
    private static Tokenizer ByteTokenizer()
    {
        var tokens = new List<string>();
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        return Tokenizer.FromGguf([.. tokens], merges: null, tokenTypes: null, bosId: -1, eosId: -1);
    }

    /// <summary>Feeds ids one at a time, as the generator does, and joins the deltas.</summary>
    private static string Stream(Tokenizer tok, int[] ids, bool final = true)
    {
        var list = new List<int>();
        int prefix = 0, read = 0;
        var output = new System.Text.StringBuilder();
        foreach (int id in ids)
        {
            list.Add(id);
            output.Append(IncrementalDetokenizer.Next(tok, list.ToArray(), ref prefix, ref read));
        }
        if (final)
            output.Append(IncrementalDetokenizer.Next(tok, list.ToArray(), ref prefix, ref read, final: true));
        return output.ToString();
    }

    [Fact]
    public void MultiByteCharacterSplitAcrossTokens_IsEmittedWhole()
    {
        var tok = ByteTokenizer();
        // "€" is E2 82 AC in UTF-8, one byte per token.
        Assert.Equal("€", Stream(tok, [0xE2, 0x82, 0xAC]));
    }

    [Fact]
    public void AsciiText_IsEmittedAsItArrives()
    {
        var tok = ByteTokenizer();
        int[] ids = "hello".Select(c => (int)c).ToArray();
        Assert.Equal("hello", Stream(tok, ids));
    }

    [Fact]
    public void MixedText_KeepsOrderAndDoesNotDuplicate()
    {
        var tok = ByteTokenizer();
        // "a€b" = 61 E2 82 AC 62
        Assert.Equal("a€b", Stream(tok, [0x61, 0xE2, 0x82, 0xAC, 0x62]));
    }

    [Fact]
    public void IncompleteTrailingBytes_AreHeldBackUntilFinal()
    {
        var tok = ByteTokenizer();
        // Only the first two bytes of "€": nothing is complete, so nothing is emitted
        // while streaming. The final flush drops the invalid remainder rather than
        // emitting U+FFFD.
        var list = new List<int>();
        int prefix = 0, read = 0;
        string streamed = "";
        foreach (int id in new[] { 0xE2, 0x82 })
        {
            list.Add(id);
            streamed += IncrementalDetokenizer.Next(tok, list.ToArray(), ref prefix, ref read);
        }
        Assert.Equal("", streamed);
        string tail = IncrementalDetokenizer.Next(tok, list.ToArray(), ref prefix, ref read, final: true);
        Assert.DoesNotContain("\uFFFD", tail);
    }
}
