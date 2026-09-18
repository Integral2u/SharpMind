using System.Diagnostics;
using System.Text;
using SharpMind.Inference.Grammar;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Grammar;

/// <summary>
/// Milestone-1 spike: verifies <see cref="TokenByteTable"/> reproduces the
/// tokenizer's own per-token rendering as raw bytes for both GPT-2 byte-level
/// BPE and SentencePiece vocabularies, and that the compact
/// <see cref="TokenTrie"/> builds and round-trips at ~150K-token scale.
/// </summary>
public sealed class TokenByteTableTests
{
    private readonly ITestOutputHelper _output;

    public TokenByteTableTests(ITestOutputHelper output) => _output = output;

    // ─── GPT-2 byte-level BPE fixture ──────────────────────────────────────

    private static readonly string[] Gpt2Tokens = BuildGpt2Tokens();
    private static readonly Tokenizer Gpt2 = Tokenizer.FromGguf(
        tokens: Gpt2Tokens, merges: null, tokenTypes: null, bosId: 1, eosId: 2);

    private static string[] BuildGpt2Tokens()
    {
        var list = new List<string> { "[UNK]", "[BOS]", "[EOS]", "[PAD]" };
        for (int b = 0; b < 256; b++)
            list.Add(Vocabulary.ByteTokenString(b));

        // Merged pieces: a space-prefixed word, a bare word, punctuation, and a
        // multi-byte UTF-8 sequence (é = 0xC3 0xA9) expressed in byte-map chars.
        list.Add(Vocabulary.ByteTokenString(0x20) + "Hello");
        list.Add("Hello");
        list.Add(Vocabulary.ByteTokenString(0x20) + "world");
        list.Add("world");
        list.Add("!");
        list.Add(Vocabulary.ByteTokenString(0xC3) + Vocabulary.ByteTokenString(0xA9));
        return [.. list];
    }

    [Fact]
    public void Gpt2_EveryOrdinaryToken_MatchesTokenizerDecodeBytes()
    {
        var table = TokenByteTable.Build(Gpt2);
        Assert.Equal(Gpt2Tokens.Length, table.Count);
        AssertMatchesDecode(Gpt2, table);
    }

    [Fact]
    public void Gpt2_KnownTokens_ExposeExpectedBytes()
    {
        var table = TokenByteTable.Build(Gpt2);

        AssertBytes(table, Gpt2.TokenToId("A"), 0x41);
        AssertBytes(table, Gpt2.TokenToId(Vocabulary.ByteTokenString(0x20)), 0x20);
        AssertBytes(table, Gpt2.TokenToId(Vocabulary.ByteTokenString(0x20) + "Hello"),
            0x20, 0x48, 0x65, 0x6C, 0x6C, 0x6F);
        AssertBytes(table, Gpt2.TokenToId(Vocabulary.ByteTokenString(0xC3) + Vocabulary.ByteTokenString(0xA9)),
            0xC3, 0xA9);
    }

    [Fact]
    public void Gpt2_SpecialTokens_AreFlaggedAndHaveNoBytes()
    {
        var table = TokenByteTable.Build(Gpt2);

        foreach (int id in new[] { Gpt2.UnkId, Gpt2.BosId, Gpt2.EosId, Gpt2.PadId })
        {
            Assert.True(table.IsSpecial(id));
            Assert.Empty(table.GetBytes(id).ToArray());
        }
    }

    // ─── SentencePiece fixture ─────────────────────────────────────────────

    private static readonly string[] SpTokens =
    [
        "<unk>", "<s>", "</s>", "<pad>",
        "\u2581", "\u2581Hello", "Hello", "H", "e", "l", "o",
        "\u2581world", "world", "<0x21>", "<0xE9>", "!",
    ];

    private static readonly float[] SpScores =
        [.. Enumerable.Repeat(-100f, 4), .. Enumerable.Repeat(0f, SpTokens.Length - 4)];

    private static readonly Tokenizer Sp = Tokenizer.FromGguf(
        tokens: SpTokens, merges: null, tokenTypes: null, bosId: 1, eosId: 2, scores: SpScores);

    [Fact]
    public void SentencePiece_EveryOrdinaryToken_MatchesTokenizerDecodeBytes()
    {
        var table = TokenByteTable.Build(Sp);
        Assert.Equal(SpTokens.Length, table.Count);
        AssertMatchesDecode(Sp, table);
    }

    [Fact]
    public void SentencePiece_Metaspace_BecomesSpaceByte()
    {
        var table = TokenByteTable.Build(Sp);

        AssertBytes(table, Sp.TokenToId("\u2581"), 0x20);
        AssertBytes(table, Sp.TokenToId("\u2581Hello"), 0x20, 0x48, 0x65, 0x6C, 0x6C, 0x6F);
    }

    [Fact]
    public void SentencePiece_ByteFallback_EmitsSingleByte()
    {
        var table = TokenByteTable.Build(Sp);

        AssertBytes(table, Sp.TokenToId("<0x21>"), 0x21);
        AssertBytes(table, Sp.TokenToId("<0xE9>"), 0xE9);
    }

    // ─── Trie ──────────────────────────────────────────────────────────────

    [Fact]
    public void Trie_RoundTripsEveryOrdinaryToken()
    {
        var table = TokenByteTable.Build(Gpt2);
        var trie = TokenTrie.Build(table);

        Span<int> ids = stackalloc int[TokenTrie.MaxTokensPerPath];
        int expected = 0;
        for (int id = 0; id < table.Count; id++)
        {
            if (table.IsSpecial(id)) continue;
            expected++;
            int count = trie.GetTokenIds(table.GetBytes(id), ids);
            Assert.True(count > 0, $"token {id} missing from trie");
            Assert.Contains(id, ids[..count].ToArray());
        }

        Assert.Equal(expected, trie.TokenCount);
    }

    [Fact]
    public void Trie_ByteSynonyms_AreAllRetained()
    {
        var table = TokenByteTable.Build(Gpt2);
        var trie = TokenTrie.Build(table);

        // The literal "!" and the single-byte "!" token share one byte sequence.
        // Byte tokens start at id 4, so byte 0x21 ('!') is id 37; the literal
        // "!" was appended after the 256 byte tokens and merged pieces.
        int byteToken = 4 + 0x21;
        int literal = Gpt2Tokens.Length - 2;
        Assert.Equal("!", Gpt2.IdToToken(byteToken));
        Assert.Equal("!", Gpt2.IdToToken(literal));

        Span<int> ids = stackalloc int[TokenTrie.MaxTokensPerPath];
        int count = trie.GetTokenIds(table.GetBytes(literal), ids);
        Assert.Equal(2, count);
        Assert.Contains(literal, ids[..count].ToArray());
        Assert.Contains(byteToken, ids[..count].ToArray());
    }

    [Fact]
    public void Trie_LongestPath_WinsOverPrefix()
    {
        var table = TokenByteTable.Build(Gpt2);
        var trie = TokenTrie.Build(table);

        // "world" and " world" both exist; the space-prefixed path must resolve
        // to its own token, not the shorter bare word.
        int bare = Gpt2.TokenToId("world");
        int spaced = Gpt2.TokenToId(Vocabulary.ByteTokenString(0x20) + "world");

        Assert.True(trie.Contains(table.GetBytes(bare), out int gotBare));
        Assert.Equal(bare, gotBare);
        Assert.True(trie.Contains(table.GetBytes(spaced), out int gotSpaced));
        Assert.Equal(spaced, gotSpaced);
    }

    // ─── Scale spike (~150K tokens) ────────────────────────────────────────

    [Fact]
    public void Scale_BuildsTableAndTrie_WithinBudget()
    {
        var tokens = BuildLargeVocab(150_000);
        var tokenizer = Tokenizer.FromGguf(tokens: tokens, merges: null, tokenTypes: null, bosId: 1, eosId: 2);

        var sw = Stopwatch.StartNew();
        var table = TokenByteTable.Build(tokenizer);
        sw.Stop();
        long tableMs = sw.ElapsedMilliseconds;

        sw.Restart();
        var trie = TokenTrie.Build(table);
        sw.Stop();
        long trieMs = sw.ElapsedMilliseconds;

        int ordinary = 0;
        for (int id = 0; id < table.Count; id++)
            if (!table.IsSpecial(id)) ordinary++;

        _output.WriteLine($"vocab={table.Count} ordinary={ordinary} trieNodes={trie.NodeCount} trieEdges={trie.EdgeCount} tableMs={tableMs} trieMs={trieMs}");

        Assert.Equal(tokens.Length, table.Count);
        Assert.Equal(ordinary, trie.TokenCount);
        Assert.True(trie.NodeCount > 0);
        Assert.True(tableMs < 10_000, $"table build took {tableMs} ms");
        Assert.True(trieMs < 10_000, $"trie build took {trieMs} ms");
    }

    // ─── Helpers ───────────────────────────────────────────────────────────

    /// <summary>
    /// The central M1 invariant: for every non-special token, UTF-8 decoding the
    /// table's bytes equals what the tokenizer's own Decode renders for that id.
    /// </summary>
    private static void AssertMatchesDecode(Tokenizer tokenizer, TokenByteTable table)
    {
        for (int id = 0; id < table.Count; id++)
        {
            if (table.IsSpecial(id)) continue;
            string fromTable = Encoding.UTF8.GetString(table.GetBytes(id));
            string fromDecode = tokenizer.Decode([id], skipSpecials: false);
            Assert.Equal(fromDecode, fromTable);
        }
    }

    private static void AssertBytes(TokenByteTable table, int id, params byte[] expected)
        => Assert.Equal(expected, table.GetBytes(id).ToArray());

    private static string[] BuildLargeVocab(int size)
    {
        var used = new HashSet<string>(capacity: size, StringComparer.Ordinal);
        var list = new List<string>(size) { "[UNK]", "[BOS]", "[EOS]", "[PAD]" };
        used.Add("[UNK]"); used.Add("[BOS]"); used.Add("[EOS]"); used.Add("[PAD]");

        for (int b = 0; b < 256; b++)
        {
            string t = Vocabulary.ByteTokenString(b);
            if (used.Add(t)) list.Add(t);
        }

        var rng = new Random(1234);
        var sb = new StringBuilder(8);
        while (list.Count < size)
        {
            int len = 2 + rng.Next(0, 7);
            sb.Clear();
            for (int i = 0; i < len; i++)
                sb.Append(Vocabulary.ByteTokenString(rng.Next(0, 256)));
            string s = sb.ToString();
            if (used.Add(s))
                list.Add(s);
        }
        return [.. list];
    }
}