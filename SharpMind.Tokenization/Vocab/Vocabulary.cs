using System.Runtime.CompilerServices;

namespace SharpMind.Tokenization.Vocab;

/// <summary>
/// Bidirectional map between token strings and integer IDs.
///
/// Special tokens occupy the lowest IDs in insertion order.
/// Byte-level fallback tokens (0x00–0xFF) come next when enabled.
/// BPE-learned merge tokens fill the remainder.
///
/// Thread-safe for reads after construction — mutation only during training.
/// </summary>
public sealed class Vocabulary
{
    private readonly Dictionary<string, int> _tokenToId;
    private readonly List<string> _idToToken;

    /// <param name="specials">Special tokens added first with the lowest IDs.</param>
    /// <param name="addByteTokens">
    /// When true, adds 256 single-byte tokens (Ġ0x00 … Ġ0xFF) after specials.
    /// Required for GPT-2 style byte-level BPE that can represent any unicode text
    /// without an [UNK] fallback.
    /// </param>
    public Vocabulary(SpecialTokens specials, bool addByteTokens = true)
    {
        Specials = specials;

        _idToToken = new List<string>(specials.All.Count + (addByteTokens ? 256 : 0) + 4096);
        _tokenToId = new Dictionary<string, int>(_idToToken.Capacity, StringComparer.Ordinal);

        foreach (string token in specials.All)
            AddToken(token);

        if (addByteTokens)
            for (int b = 0; b < 256; b++)
                AddToken(ByteTokenString(b));
    }

    /// <summary>Constructs from a pre-built list (used when loading from disk).</summary>
    internal Vocabulary(IReadOnlyList<string> tokens, SpecialTokens specials)
    {
        Specials = specials;
        _idToToken = [.. tokens];
        _tokenToId = new Dictionary<string, int>(tokens.Count, StringComparer.Ordinal);
        for (int i = 0; i < tokens.Count; i++)
            _tokenToId[tokens[i]] = i;
    }

    public SpecialTokens Specials { get; }
    public int Size => _idToToken.Count;
    public int UnkId => _tokenToId.TryGetValue(Specials.Unk, out int unk) ? unk : 0;
    public int BosId => _tokenToId.TryGetValue(Specials.Bos, out int bos) ? bos : 1;
    public int EosId => _tokenToId.TryGetValue(Specials.Eos, out int eos) ? eos : 2;
    public int PadId => _tokenToId.TryGetValue(Specials.Pad, out int pad) ? pad : 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetId(string token, out int id)
        => _tokenToId.TryGetValue(token, out id);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetId(string token)
        => _tokenToId.TryGetValue(token, out int id) ? id : UnkId;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public string GetToken(int id)
        => (uint)id < (uint)_idToToken.Count ? _idToToken[id] : Specials.Unk;

    public bool Contains(string token) => _tokenToId.ContainsKey(token);

    // Mutation (training only)

    /// <summary>
    /// Adds a new token and returns its assigned ID.
    /// No-ops and returns the existing ID if the token is already present.
    /// </summary>
    public int AddToken(string token)
    {
        if (_tokenToId.TryGetValue(token, out int existing))
            return existing;

        int id = _idToToken.Count;
        _idToToken.Add(token);
        _tokenToId[token] = id;
        return id;
    }

    public IReadOnlyList<string> AllTokens => _idToToken;

    /// <summary>
    /// Returns the canonical byte token string for byte value <paramref name="b"/>.
    /// Uses the GPT-2 / LLaMA byte-level BPE mapping to avoid control characters.
    /// </summary>
    public static string ByteTokenString(int b)
    {
        if (b < 0 || b > 255) return $"<0x{b:X2}>";

        // The official GPT-2 byte-to-unicode mapping
        // This map ensures that all 256 bytes are mapped to printable characters.
        // This is critical for BPE merge rules to match correctly.
        return ByteMap[b];
    }

    private static readonly string[] ByteMap = CreateByteMap();

    private static string[] CreateByteMap()
    {
        // Exact OpenAI GPT-2 byte-to-unicode mapping from
        // https://github.com/openai/gpt-2/blob/master/src/encoder.py
        var bs = new List<int>(256);
        bs.AddRange(Enumerable.Range(33, 94));   // 33-126
        bs.AddRange(Enumerable.Range(161, 12));  // 161-172
        bs.AddRange(Enumerable.Range(174, 82));  // 174-255

        var cs = new List<int>(bs);

        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!bs.Contains(b))
            {
                bs.Add(b);
                cs.Add(256 + n);
                n++;
            }
        }

        var map = new string[256];
        for (int i = 0; i < 256; i++)
            map[bs[i]] = char.ConvertFromUtf32(cs[i]);
        return map;
    }

    private static readonly Dictionary<char, byte> ReverseByteMap = CreateReverseByteMap();

    private static Dictionary<char, byte> CreateReverseByteMap()
    {
        var map = new Dictionary<char, byte>(256);
        for (int b = 0; b < 256; b++)
        {
            string s = ByteMap[b];
            if (s.Length == 1)
                map[s[0]] = (byte)b;
        }
        return map;
    }

    internal static bool TryDecodeByteToken(string token, out byte b)
    {
        if (token.Length == 1 && ReverseByteMap.TryGetValue(token[0], out b))
            return true;
        if (token.StartsWith("<0x", StringComparison.Ordinal) && token.EndsWith('>') &&
            token.Length == 6 &&
            byte.TryParse(token[3..5], System.Globalization.NumberStyles.HexNumber, null, out b))
            return true;
        b = 0;
        return false;
    }

    /// <summary>Encodes a string to byte tokens using the byte-level fallback.</summary>
    public static IEnumerable<string> ByteTokenise(string text)
    {
        foreach (byte b in System.Text.Encoding.UTF8.GetBytes(text))
            yield return ByteTokenString(b);
    }
}
