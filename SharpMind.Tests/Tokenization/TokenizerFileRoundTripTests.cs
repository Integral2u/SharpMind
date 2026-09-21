using SharpMind.Tokenization;
using Xunit;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// Round-trip tests for <c>TokenizerFile</c>. The key regression under test:
/// SentencePiece-style models rank merges by per-token score
/// (<c>tokenizer.ggml.scores</c>), so a save/load that silently dropped the
/// scores would produce different encodings after reload.
/// </summary>
public sealed class TokenizerFileRoundTripTests
{
    private const string Unk = "<unk>";
    private const string Bos = "<s>";
    private const string Eos = "</s>";

    // Score setup that makes merge priority depend on the score values:
    //   "ab" has a low score, "bx" has a high score.
    // For input "abx" the pair (b,x)="bx" must win over (a,b)="ab", giving
    // [▁, a, bx] → ids [3, 4, 8]. If scores are lost (all zero), the two
    // pairs tie and the result is not the score-ranked one.
    private static readonly string[] Vocab = [Unk, Bos, Eos, "\u2581", "a", "b", "x", "ab", "bx"];
    private static readonly float[] Scores = [0f, 0f, 0f, 0f, 0f, 0f, 0f, 5f, 50f];

    [Fact]
    public void Json_RoundTrip_PreservesScores_AndEncoding()
    {
        var tok = Tokenizer.FromGguf(
            tokens: Vocab,
            merges: null,
            tokenTypes: null,
            bosId: 1,
            eosId: 2,
            scores: Scores);

        int[] expected = [3, 4, 8];
        Assert.Equal(expected, tok.Encode("abx", addBos: false, addEos: false));

        string json = tok.ToJson();
        Assert.Contains("\"scores\"", json);

        var reloaded = Tokenizer.FromJson(json);
        Assert.Equal(expected, reloaded.Encode("abx", addBos: false, addEos: false));
    }

    [Fact]
    public void Json_RoundTrip_WithoutScores_OmitsScoresKey()
    {
        // Character-level and plain BPE tokenizers have no scores; the key must
        // be absent so legacy files keep loading with scores == null.
        var tok = Tokenizer.FromGguf(
            tokens: Vocab,
            merges: ["a b", "ab x"],
            tokenTypes: null,
            bosId: 1,
            eosId: 2);

        string json = tok.ToJson();
        Assert.DoesNotContain("\"scores\"", json);

        var reloaded = Tokenizer.FromJson(json);
        Assert.Equal(9, reloaded.VocabSize);
    }
}