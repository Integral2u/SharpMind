using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using Xunit;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// BPE must apply every merge the vocabulary supports, not only those adjacent
/// to the last merge point.
///
/// The merge loop used to hold a priority queue of *positions* into a list it
/// mutated with RemoveAt. A merge shifted every later element down one, so a
/// queued position further right then referred to the wrong pair, failed
/// revalidation, and was discarded. Only positions idx-1/idx/idx+1 were
/// re-queued, so a pending merge separated from the merge point by
/// unmergeable tokens was lost for good.
///
/// Against a real Qwen2 vocabulary that shredded 9 of 20 common words
/// (" helpful" -> "Ġhelp|fu|l", " assistant" -> "Ġass|is|ta|nt") while
/// decode(encode(x)) still round-tripped, which is why it went unnoticed.
/// </summary>
public sealed class BpeMergeCompletenessTests
{
    /// <summary>UNK/BOS/EOS, the 256 GPT-2 byte tokens, then the merge results.</summary>
    private static Tokenizer BuildTokenizer(string[] extraTokens, string[] merges)
    {
        var tokens = new List<string> { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++) tokens.Add(Vocabulary.ByteTokenString(b));
        tokens.AddRange(extraTokens);
        return Tokenizer.FromGguf([.. tokens], merges, tokenTypes: null, bosId: 1, eosId: 2);
    }

    /// <summary>
    /// "abzzzef": (a,b) merges at the far left and (e,f) at the far right, with
    /// three unmergeable tokens between them so no merge is ever adjacent to
    /// the (e,f) pair. Merging (a,b) shifts (e,f) left by one; the old code
    /// dropped its stale queue entry and never revisited it, emitting
    /// "ab|z|z|z|e|f" instead of "ab|z|z|z|ef".
    /// </summary>
    [Fact]
    public void MergeSeparatedFromMergePoint_IsNotDropped()
    {
        var tok = BuildTokenizer(["ab", "ef"], ["a b", "e f"]);

        int[] ids = tok.Encode("abzzzef", addBos: false, addEos: false);

        Assert.Equal(["ab", "z", "z", "z", "ef"], [.. ids.Select(tok.IdToToken)]);
    }

    /// <summary>Same shape, with the distant merge outranking the near one.</summary>
    [Fact]
    public void DistantMerge_SurvivesRegardlessOfRankOrder()
    {
        var tok = BuildTokenizer(["ab", "ef"], ["e f", "a b"]);

        int[] ids = tok.Encode("abzzzef", addBos: false, addEos: false);

        Assert.Equal(["ab", "z", "z", "z", "ef"], [.. ids.Select(tok.IdToToken)]);
    }

    /// <summary>A full merge chain still collapses to the single largest token.</summary>
    [Fact]
    public void FullMergeChain_CollapsesToOneToken()
    {
        var tok = BuildTokenizer(
            ["ab", "cd", "ef", "gh", "abcd", "efgh", "abcdefgh"],
            ["g h", "a b", "e f", "c d", "ab cd", "ef gh", "abcd efgh"]);

        int[] ids = tok.Encode("abcdefgh", addBos: false, addEos: false);

        Assert.Single(ids);
        Assert.Equal("abcdefgh", tok.IdToToken(ids[0]));
    }

    [Fact]
    public void Encode_RoundTripsAfterMerging()
    {
        var tok = BuildTokenizer(["ab", "ef"], ["a b", "e f"]);

        foreach (string text in new[] { "abzzzef", "ab", "ef", "zzz" })
            Assert.Equal(text, tok.Decode(tok.Encode(text, addBos: false, addEos: false)));
    }
}
