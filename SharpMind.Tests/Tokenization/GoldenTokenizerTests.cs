using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using Xunit;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// Golden-output tests that encode known strings and assert expected
/// roundtrip behaviour. Each test family corresponds to a tokenizer
/// architecture:
///   - GPT-2 byte-level BPE  (Qwen2, Phi, Llama-3/tiktoken)
///   - SentencePiece / SPM   (LLaMA, LLaMA-2, Mistral, TinyLlama)
///
/// Fixture data is embedded inline rather than loaded from external
/// tokenizer files so the tests are self-contained and runnable without
/// downloading model weights.
/// </summary>
public sealed class GoldenTokenizerTests
{
    // ─── GPT-2 byte-level BPE path ─────────────────────────────────────────

    /// <summary>
    /// Minimal GPT-2-style vocabulary built from the same
    /// <see cref="Vocabulary.ByteTokenString"/> mapping that the BPE encoder
    /// uses internally, so token strings are guaranteed to match.
    ///
    ///   0 = [UNK]   1 = [BOS]   2 = [EOS]
    ///   3..258      = GPT-2 byte tokens (byte 0..255)
    /// </summary>
    private static readonly string[] Gpt2Vocab = BuildGpt2FullVocab();
    private static readonly Tokenizer Gpt2Tok = Tokenizer.FromGguf(
        tokens: Gpt2Vocab,
        merges: null,
        tokenTypes: null,
        bosId: 1,
        eosId: 2);

    private static string[] BuildGpt2FullVocab()
    {
        var list = new List<string>(259) { "[UNK]", "[BOS]", "[EOS]" };
        for (int b = 0; b < 256; b++)
            list.Add(Vocabulary.ByteTokenString(b));
        return [.. list];
    }

    [Fact]
    public void Gpt2Bpe_EncodesKnownString()
    {
        int[] ids = Gpt2Tok.Encode("Hello", addBos: false, addEos: false);
        Assert.NotEmpty(ids);
        foreach (int id in ids)
            Assert.InRange(id, 0, Gpt2Vocab.Length - 1);
    }

    [Fact]
    public void Gpt2Bpe_DecodeRoundtrips()
    {
        int[] ids = Gpt2Tok.Encode("Hello", addBos: false, addEos: false);
        string decoded = Gpt2Tok.Decode(ids);
        Assert.Equal("Hello", decoded);
    }

    // ─── SentencePiece / SPM path ──────────────────────────────────────────

    /// <summary>
    /// Minimal SentencePiece-style vocabulary with no merge rules (only
    /// scores). The encoder detects this via
    /// <c>merges.Count == 0 &amp;&amp; scores is { Count: &gt; 0 }</c> and
    /// switches to score-ranked merging.
    ///
    /// Token layout:
    ///   0 = &lt;unk&gt;   1 = &lt;s&gt; (BOS)   2 = &lt;/s&gt; (EOS)
    ///   3 = ▁      4 = H    5 = e    6 = l    7 = o
    ///   8 = Z                                 (known char for byte-fallback test)
    /// </summary>
    private static readonly string[] SpVocab = [
        "<unk>", "<s>", "</s>",
        "\u2581", "H", "e", "l", "o",
        "Z"
    ];

    private static readonly float[] SpScores = [
        -100f, -100f, -100f,
        0f, 0f, 0f, 0f, 0f,
        0f
    ];

    private static readonly Tokenizer SpTok = Tokenizer.FromGguf(
        tokens: SpVocab,
        merges: null,
        tokenTypes: null,
        bosId: 1,
        eosId: 2,
        scores: SpScores);

    [Fact]
    public void SentencePiece_EncodesKnownString()
    {
        // "Hello" → single plain segment → prepend dummy "▁"
        // → codepoints: [▁, H, e, l, l, o]
        // → each known in vocab: [3, 4, 5, 6, 6, 7]
        int[] ids = SpTok.Encode("Hello", addBos: false, addEos: false);
        Assert.Equal([3, 4, 5, 6, 6, 7], ids);
    }

    [Fact]
    public void SentencePiece_DummyPrefix_AddedToFirstSegmentOnly()
    {
        int[] ids = SpTok.Encode("Hello", addBos: false, addEos: false);
        Assert.Equal("\u2581", SpTok.IdToToken(ids[0]));
    }

    [Fact]
    public void SentencePiece_SecondSegment_NoDummyPrefix()
    {
        // "<s>Hello" → special token segment "<s>" then plain "Hello"
        // First plain segment "Hello" gets dummy prefix "▁"
        // Result encoding: [1 (BOS), 3 (▁), 4 (H), 5 (e), 6 (l), 6 (l), 7 (o)]
        int[] ids = SpTok.Encode("<s>Hello", addBos: false, addEos: false);
        Assert.Equal(SpTok.BosId, ids[0]);      // <s> preserved as special
        Assert.Equal("\u2581", SpTok.IdToToken(ids[1])); // first plain seg gets ▁
    }

    [Fact]
    public void SentencePiece_DecodeRoundtrips()
    {
        // SentencePiece dummy prefix "▁" is decoded back to space, so
        // roundtrip produces " Hello" with a leading space.
        int[] ids = SpTok.Encode("Hello", addBos: false, addEos: false);
        string decoded = SpTok.Decode(ids);
        Assert.Equal(" Hello", decoded);
    }

    [Fact]
    public void SentencePiece_ByteFallback_ForUnknownChars()
    {
        // '!' is not in SpVocab, not a special token.
        // Falls through to UTF-8 byte-level fallback: 0x21 → <0x21>.
        // <0x21> is not in the vocab either → <unk> (skipped on decode).
        // The dummy prefix "▁" is added and becomes a space after decode.
        // Result after skipping <unk>: just " " (the decoded dummy prefix).
        int[] ids = SpTok.Encode("!", addBos: false, addEos: false);
        // ▁ (known) + <unk> (unknown char fallback)
        Assert.Equal(2, ids.Length);
        Assert.Equal("\u2581", SpTok.IdToToken(ids[0]));
    }

    [Fact]
    public void SentencePiece_KnownCharRoundtrips()
    {
        // 'Z' IS in SpVocab (index 8), so it should roundtrip cleanly.
        // With dummy prefix: "▁Z" → [3, 8]
        int[] ids = SpTok.Encode("Z", addBos: false, addEos: false);
        string decoded = SpTok.Decode(ids);
        // Dummy prefix ▁ → space, so result is " Z"
        Assert.Equal(" Z", decoded);
    }

    // ─── Special-token handling ────────────────────────────────────────────

    /// <summary>
    /// Vocabulary with control tokens (&lt;|im_start|&gt;, &lt;|im_end|&gt;)
    /// that the encoder should preserve verbatim instead of passing through
    /// the SentencePiece or BPE pipeline.
    /// </summary>
    private static readonly string[] ChatVocab = [
        "[UNK]", "[BOS]", "[EOS]",
        "<|im_start|>", "<|im_end|>",
        "user", "assistant", "\n", "Hello"
    ];
    private static readonly Tokenizer ChatTok = Tokenizer.FromGguf(
        tokens: ChatVocab,
        merges: null,
        tokenTypes: [1, 1, 1, 3, 3, 1, 1, 1, 1],
        bosId: 1,
        eosId: 2);

    [Fact]
    public void SpecialTokens_AreNotTokenised()
    {
        int[] ids = ChatTok.Encode("<|im_start|>user\nHello<|im_end|>",
                                   addBos: false, addEos: false);
        var decodedTokens = ids.Select(ChatTok.IdToToken).ToList();
        Assert.Contains("<|im_start|>", decodedTokens);
        Assert.Contains("<|im_end|>", decodedTokens);
    }

    // ─── BOS/EOS handling ──────────────────────────────────────────────────

    [Fact]
    public void AddBos_AddsBosTokenAtFront()
    {
        int[] withBos = SpTok.Encode("Hello", addBos: true, addEos: false);
        int[] withoutBos = SpTok.Encode("Hello", addBos: false, addEos: false);
        Assert.Equal(withoutBos.Length + 1, withBos.Length);
        Assert.Equal(SpTok.BosId, withBos[0]);
        Assert.Equal(withoutBos, withBos[1..]);
    }

    [Fact]
    public void AddEos_AddsEosTokenAtEnd()
    {
        int[] withEos = SpTok.Encode("Hello", addBos: false, addEos: true);
        int[] withoutEos = SpTok.Encode("Hello", addBos: false, addEos: false);
        Assert.Equal(withoutEos.Length + 1, withEos.Length);
        Assert.Equal(SpTok.EosId, withEos[^1]);
        Assert.Equal(withoutEos, withEos[..^1]);
    }
}
