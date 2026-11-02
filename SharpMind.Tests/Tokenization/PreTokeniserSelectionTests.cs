using SharpMind.Tokenization;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Serialisation;
using SharpMind.Tokenization.Vocab;
using Xunit;

namespace SharpMind.Tests.Tokenization;

/// <summary>
/// Which pre-tokeniser a tokenizer gets, and what each one splits. The llama.cpp golden fixtures
/// (<see cref="LlamaCppVocabParityTests"/>) prove the end result for four vocabularies; these pin the
/// pieces they cannot isolate: pre name winning over architecture, the architecture fallback,
/// saved-file names, the HF converters, and the merge tie order on a vocab built for it.
/// </summary>
public sealed class PreTokeniserSelectionTests
{
    // ── What each pattern splits ───────────────────────────────────────────

    [Theory]
    [InlineData(Cl100kVariant.Qwen2, "ns.Core;\nx", new[] { "ns", ".Core", ";\n", "x" })]
    [InlineData(Cl100kVariant.Qwen2, "a\n\n   b", new[] { "a", "\n\n", "  ", " b" })]
    [InlineData(Cl100kVariant.Qwen2, "x 12345", new[] { "x", " ", "1", "2", "3", "4", "5" })]
    [InlineData(Cl100kVariant.Llama3, "x 12345", new[] { "x", " ", "123", "45" })]
    [InlineData(Cl100kVariant.Cl100k, "x 12345", new[] { "x", " ", "123", "45" })]
    [InlineData(Cl100kVariant.Qwen2, "I'M here", new[] { "I", "'M", " here" })]
    // Combining acute accent (U+0301): Qwen2 leaves it outside the letter run, Qwen3.5 keeps it in.
    [InlineData(Cl100kVariant.Qwen2, "éx", new[] { "e", "́x" })]
    [InlineData(Cl100kVariant.Qwen35, "éx", new[] { "éx" })]
    public void Variant_SplitsLikeLlamaCpp(Cl100kVariant variant, string text, string[] expected)
        => Assert.Equal(expected, new Cl100kPreTokeniser(variant).PreTokenise(text).ToArray());

    [Fact]
    public void OnlyLlama3_IgnoresMerges()
    {
        Assert.True(new Cl100kPreTokeniser(Cl100kVariant.Llama3).IgnoreMerges);
        Assert.False(new Cl100kPreTokeniser(Cl100kVariant.Cl100k).IgnoreMerges);
        Assert.False(new Cl100kPreTokeniser(Cl100kVariant.Qwen2).IgnoreMerges);
        Assert.False(new Cl100kPreTokeniser(Cl100kVariant.Qwen35).IgnoreMerges);
    }

    // ── Selection: tokenizer.ggml.pre first, then general.architecture ─────

    [Theory]
    // The pre name wins over a conflicting architecture.
    [InlineData("llama", "qwen2", "qwen2")]
    [InlineData("qwen2", "llama-bpe", "llama3")]
    [InlineData("qwen2", "gpt-2", "gpt2")]
    [InlineData("llama", "dbrx", "cl100k")]
    [InlineData("qwen3", "qwen35", "qwen35")]
    [InlineData("llama", "hunyuan", "qwen2")]
    [InlineData("llama", "chatglm-bpe", "cl100k")]
    [InlineData("llama", "olmo", "gpt2")]
    // Absent or unknown pre name: architecture fallback.
    [InlineData("qwen35moe", null, "qwen35")]
    [InlineData("qwen3", null, "qwen2")]
    [InlineData("qwen2", "not-a-real-pre", "qwen2")]
    [InlineData("llama", null, "llama3")]
    [InlineData("lfm2", null, "llama3")]
    [InlineData("mistral3", "tekken", "qwen2")]
    [InlineData("dbrx", null, "cl100k")]
    [InlineData("deepseek2", null, "cl100k")]
    [InlineData("gpt2", null, "gpt2")]
    [InlineData(null, null, "gpt2")]
    public void Gguf_SelectsPreTokeniser(string? arch, string? pre, string expectedName)
    {
        var model = GgufConverter.Convert(ByteVocab(), ["a b"], null, -1, -1, architecture: arch, preTokenizer: pre);

        Assert.Equal(expectedName, Describe(model.PreTokeniser));
        Assert.Equal(pre, model.GgufPreTokenizer);
    }

    private static string Describe(IPreTokeniser pt) => pt switch
    {
        Gpt2PreTokeniser => "gpt2",
        Cl100kPreTokeniser c => c.Variant.ToString().ToLowerInvariant(),
        _ => pt.GetType().Name,
    };

    [Fact]
    public void Tokenizer_FromGguf_PassesPreName()
    {
        // Through the public entry point GgufLoader uses: llama arch, qwen2 pre -> single digits.
        var tok = Tokenizer.FromGguf(ByteVocab(), ["a b"], null, -1, -1, architecture: "llama", preTokenizer: "qwen2");

        Assert.Equal(5, tok.Encode("12345", addBos: false, addEos: false).Length);
        Assert.Equal("qwen2", tok.GgufPreTokenizer);
    }

    // ── Saved tokenizer files ──────────────────────────────────────────────

    [Theory]
    [InlineData("llama-bpe", Cl100kVariant.Llama3, "llama3")]
    [InlineData("qwen2", Cl100kVariant.Qwen2, "qwen2")]
    [InlineData("qwen35", Cl100kVariant.Qwen35, "qwen35")]
    [InlineData("dbrx", Cl100kVariant.Cl100k, "cl100k")]
    public void TokenizerFile_RoundTripsVariant(string pre, Cl100kVariant variant, string name)
    {
        var model = GgufConverter.Convert(ByteVocab(), ["a b"], null, -1, -1, preTokenizer: pre);
        model.GgufPreTokenizer = null; // the variant key alone must carry it
        string json = TokenizerFile.ToJson(model);
        // "pre_tokeniser" stays "cl100k" so older SharpMind readers still load a cl100k-family pre-tokeniser.
        Assert.Contains("\"pre_tokeniser\": \"cl100k\"", json);
        Assert.Contains($"\"cl100k_variant\": \"{name}\"", json);

        var reloaded = TokenizerFile.FromJson(json);
        Assert.Equal(variant, Assert.IsType<Cl100kPreTokeniser>(reloaded.PreTokeniser).Variant);
    }

    [Theory]
    // Files written before the variants existed said "cl100k" for every tiktoken model.
    [InlineData("qwen2", Cl100kVariant.Qwen2)]
    [InlineData("llama-bpe", Cl100kVariant.Llama3)]
    [InlineData("qwen35", Cl100kVariant.Qwen35)]
    [InlineData(null, Cl100kVariant.Cl100k)]
    [InlineData("gpt-2", Cl100kVariant.Cl100k)] // a non-tiktoken pre name cannot turn "cl100k" into GPT-2
    public void TokenizerFile_LegacyCl100k_UsesStoredGgufPre(string? ggufPre, Cl100kVariant expected)
    {
        var model = GgufConverter.Convert(ByteVocab(), ["a b"], null, -1, -1, architecture: "dbrx");
        model.GgufPreTokenizer = ggufPre;
        // An older writer: same file without the variant key.
        var obj = System.Text.Json.Nodes.JsonNode.Parse(TokenizerFile.ToJson(model))!.AsObject();
        Assert.True(obj.Remove("cl100k_variant"));
        Assert.Equal("cl100k", (string?)obj["pre_tokeniser"]);

        var reloaded = TokenizerFile.FromJson(obj.ToJsonString());
        Assert.Equal(expected, Assert.IsType<Cl100kPreTokeniser>(reloaded.PreTokeniser).Variant);
    }

    // ── HF tokenizer.json converters ───────────────────────────────────────

    [Fact]
    public void QwenConverter_UsesQwen2()
    {
        string path = WriteTokenizerJson("""{"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]}}""");
        try { Assert.Equal(Cl100kVariant.Qwen2, Assert.IsType<Cl100kPreTokeniser>(QwenConverter.Convert(path).PreTokeniser).Variant); }
        finally { File.Delete(path); }
    }

    [Fact]
    public void LlamaConverter_Llama3_UsesLlama3()
    {
        string path = WriteTokenizerJson(
            """{"added_tokens":[{"id":2,"content":"<|begin_of_text|>","special":true}],"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":["a b"]}}""");
        try { Assert.Equal(Cl100kVariant.Llama3, Assert.IsType<Cl100kPreTokeniser>(LlamaConverter.Convert(path).PreTokeniser).Variant); }
        finally { File.Delete(path); }
    }

    // ── Merge tie order ────────────────────────────────────────────────────

    [Fact]
    public void ByteBpe_EqualRankPairs_MergeLeftmostFirst()
    {
        // Eleven spaces with merges that double a space run: leftmost-first BPE gives 8 + 2 + 1.
        // Any other order between the equal-rank (space, space) pairs leaves different fragments.
        string sp = Vocabulary.ByteTokenString(' ');
        string sp2 = sp + sp, sp4 = sp2 + sp2, sp8 = sp4 + sp4;
        var tokens = ByteVocab().Concat([sp2, sp4, sp8]).ToArray();
        var tok = Tokenizer.FromGguf(tokens, [$"{sp} {sp}", $"{sp2} {sp2}", $"{sp4} {sp4}"], null, -1, -1, architecture: "gpt2");

        int[] ids = tok.Encode(new string(' ', 11), addBos: false, addEos: false);

        int[] expected = [Array.IndexOf(tokens, sp8), Array.IndexOf(tokens, sp2), Array.IndexOf(tokens, sp)];
        Assert.Equal(expected, ids);
    }

    [Fact]
    public void SentencePiece_EqualScorePairs_MergeLeftmostFirst()
    {
        // No merges -> score-ranked SentencePiece path. "aaa" offers (a,a) twice at the same score:
        // leftmost-first gives ▁ aa a, the other order ▁ a aa.
        string[] tokens = ["<unk>", "<s>", "</s>", "▁", "a", "aa"];
        float[] scores = [0f, 0f, 0f, 0f, 0f, 1f];
        var tok = Tokenizer.FromGguf(tokens, merges: null, tokenTypes: null, bosId: 1, eosId: 2, scores: scores);

        Assert.Equal([3, 5, 4], tok.Encode("aaa", addBos: false, addEos: false));
    }

    [Theory]
    // Vocab has "abc" but the only merge is a+b: Llama 3 (ignore_merges) takes the whole word,
    // the others merge to "ab" + "c".
    [InlineData("llama-bpe", new[] { "abc" })]
    [InlineData("dbrx", new[] { "ab", "c" })]
    [InlineData("qwen2", new[] { "ab", "c" })]
    public void IgnoreMerges_OnlyForLlama3(string pre, string[] expectedTokens)
    {
        var tokens = ByteVocab().Concat(["ab", "abc"]).ToArray();
        var tok = Tokenizer.FromGguf(tokens, ["a b"], null, -1, -1, architecture: "llama", preTokenizer: pre);

        int[] expected = [.. expectedTokens.Select(t => Array.IndexOf(tokens, t))];
        Assert.Equal(expected, tok.Encode("abc", addBos: false, addEos: false));
    }

    private static string[] ByteVocab()
    {
        var tokens = new string[256];
        for (int b = 0; b < 256; b++) tokens[b] = Vocabulary.ByteTokenString(b);
        return tokens;
    }

    private static string WriteTokenizerJson(string json)
    {
        string path = Path.Combine(Path.GetTempPath(), $"sm-tok-{Guid.NewGuid():N}.json");
        File.WriteAllText(path, json);
        return path;
    }
}
