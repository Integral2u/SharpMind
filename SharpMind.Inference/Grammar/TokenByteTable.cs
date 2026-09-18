using System.Runtime.CompilerServices;
using System.Text;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Inference.Grammar;

/// <summary>
/// The raw byte sequence each vocabulary token contributes when emitted,
/// computed once and cached per <see cref="Tokenizer"/>.
///
/// A constrained decoder models the byte stream the model produces, so it
/// needs the exact bytes each token writes. Rendering mirrors how the
/// tokenizer's own decode presents a token as text:
///   - SentencePiece vocabularies (any token contains ▁ U+2581): the text is
///     UTF-8 with ▁ replaced by a space — the same word-boundary marker
///     mapping the encoder applies. Every token takes this path when the
///     vocab is SentencePiece-style, so accented pieces are never mistaken
///     for GPT-2 byte-map characters.
///   - GPT-2 byte-level BPE: each character reverses through the GPT-2
///     byte→char map used at training time (see <see cref="Vocabulary.ByteTokenString"/>).
///   - <c>&lt;0xNN&gt;</c> SentencePiece byte fallbacks emit the single byte.
///   - Anything else (e.g. char-mode tokenizers) is UTF-8 verbatim.
///
/// The one deliberate difference to <see cref="Tokenizer.Decode"/> is that
/// ▁ maps to a space byte (0x20) rather than its own UTF-8 sequence, because
/// generation constraints reason about the text the user sees — llama.cpp
/// applies the same ▁→space mapping when it materialises each piece.
///
/// Special-token ids are flagged (their bytes are not table entries); the
/// caller decides separately whether the grammar may emit them (EOS gating).
/// </summary>
public sealed class TokenByteTable
{
    private static readonly ConditionalWeakTable<Tokenizer, TokenByteTable> Cache = new();

    private readonly byte[][] _bytes;
    private readonly bool[] _special;

    /// <summary>Returns the per-tokenizer table, cached for reuse across sessions.</summary>
    public static TokenByteTable Get(Tokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        return Cache.GetValue(tokenizer, static t => new TokenByteTable(t));
    }

    /// <summary>Builds a fresh, uncached table (used by the perf spike tests).</summary>
    public static TokenByteTable Build(Tokenizer tokenizer) => new(tokenizer);

    /// <summary>The number of token ids covered (always <see cref="Tokenizer.VocabSize"/>).</summary>
    public int Count => _bytes.Length;

    /// <summary>True when the id is a special token (not an ordinary generation token).</summary>
    public bool IsSpecial(int id)
        => (uint)id < (uint)_special.Length && _special[id];

    /// <summary>The byte sequence this token writes, or empty for special/unknown ids.</summary>
    public ReadOnlySpan<byte> GetBytes(int id)
        => (uint)id < (uint)_bytes.Length ? _bytes[id] : default;

    private TokenByteTable(Tokenizer tokenizer)
    {
        int size = tokenizer.VocabSize;
        _bytes = new byte[size][];
        _special = new bool[size];

        var specials = new HashSet<string>(tokenizer.Specials.All, StringComparer.Ordinal);
        bool spMode = AnyMetaspace(tokenizer);

        for (int id = 0; id < size; id++)
        {
            string token = tokenizer.IdToToken(id);
            bool isSpecial = specials.Contains(token);
            _special[id] = isSpecial;
            _bytes[id] = isSpecial ? [] : ToBytes(token, spMode);
        }
    }

    /// <summary>
    /// True when any vocab token contains the SentencePiece metaspace marker ▁,
    /// identifying SentencePiece-style vocabularies (same detection the BPE
    /// encoder uses to pick its merge strategy).
    /// </summary>
    private static bool AnyMetaspace(Tokenizer tokenizer)
    {
        foreach (string token in tokenizer.Vocab.AllTokens)
            if (token.Contains('\u2581'))
                return true;
        return false;
    }

    private static byte[] ToBytes(string token, bool spMode)
    {
        // <0xNN> SentencePiece byte fallback: emits a single raw byte.
        if (token.Length == 6
            && token[0] == '<' && token[1] == '0' && token[2] == 'x'
            && token[5] == '>'
            && byte.TryParse(token.AsSpan(3, 2), System.Globalization.NumberStyles.HexNumber, null, out byte hexByte))
            return [hexByte];

        if (spMode)
            return Encoding.UTF8.GetBytes(token.Replace('\u2581', ' '));

        var bytes = new byte[token.Length];
        for (int i = 0; i < token.Length; i++)
        {
            if (!ReverseByteMap.TryGetValue(token[i], out byte b))
                return Encoding.UTF8.GetBytes(token);
            bytes[i] = b;
        }
        return bytes;
    }

    /// <summary>Recreates the GPT-2 byte→char map's inverse from the public API.</summary>
    private static readonly Dictionary<char, byte> ReverseByteMap = CreateReverseByteMap();

    private static Dictionary<char, byte> CreateReverseByteMap()
    {
        var map = new Dictionary<char, byte>(256);
        for (int b = 0; b < 256; b++)
        {
            string s = Vocabulary.ByteTokenString(b);
            if (s.Length == 1)
                map[s[0]] = (byte)b;
        }
        return map;
    }
}