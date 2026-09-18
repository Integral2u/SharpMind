using SharpMind.Inference.Grammar;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tests.Grammar;

/// <summary>
/// A byte-level test tokenizer whose vocabulary contains exactly one token per
/// byte value (ids 4..259), so grammar constraints can be driven one byte at a
/// time and every byte of a UTF-8 stream maps to a known token id.
/// </summary>
internal static class TestTokenizer
{
    public static readonly string[] Tokens = BuildTokens();
    public static readonly Tokenizer Instance = Tokenizer.FromGguf(
        tokens: Tokens, merges: null, tokenTypes: null, bosId: 1, eosId: 2);
    public static readonly TokenByteTable ByteTable = TokenByteTable.Build(Instance);

    public static int EosId => Instance.EosId;
    public static int Count => ByteTable.Count;

    /// <summary>The token id that emits <paramref name="b"/>.</summary>
    public static int IdFor(byte b) => 4 + b;

    public static string ByteToken(byte b) => Vocabulary.ByteTokenString(b);

    private static string[] BuildTokens()
    {
        var list = new List<string> { "[UNK]", "[BOS]", "[EOS]", "[PAD]" };
        for (int b = 0; b < 256; b++)
            list.Add(Vocabulary.ByteTokenString(b));
        return [.. list];
    }
}