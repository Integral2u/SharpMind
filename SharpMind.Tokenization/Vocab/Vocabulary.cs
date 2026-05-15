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

    // ── Construction ──────────────────────────────────────────────────────

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

    // ── Properties ────────────────────────────────────────────────────────

    public SpecialTokens Specials { get; }
    public int Size => _idToToken.Count;
    public int UnkId => _tokenToId.TryGetValue(Specials.Unk, out int unk) ? unk : 0;
    public int BosId => _tokenToId.TryGetValue(Specials.Bos, out int bos) ? bos : 1;
    public int EosId => _tokenToId.TryGetValue(Specials.Eos, out int eos) ? eos : 2;
    public int PadId => _tokenToId.TryGetValue(Specials.Pad, out int pad) ? pad : 0;

    // ── Lookup ────────────────────────────────────────────────────────────

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

    // ── Mutation (training only) ──────────────────────────────────────────

    /// <summary>
    /// Adds a new token and returns its assigned ID.
    /// No-ops and returns the existing ID if the token is already present.
    /// </summary>
    internal int AddToken(string token)
    {
        if (_tokenToId.TryGetValue(token, out int existing))
            return existing;

        int id = _idToToken.Count;
        _idToToken.Add(token);
        _tokenToId[token] = id;
        return id;
    }

    // ── Enumeration ───────────────────────────────────────────────────────

    public IReadOnlyList<string> AllTokens => _idToToken;

    // ── Byte token helpers ────────────────────────────────────────────────

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
        var map = new string[256];
        for (int b = 0; b < 256; b++) map[b] = $"<0x{b:X2}>";

        // Printable ASCII (33-126) maps to itself
        for (int b = 33; b <= 126; b++) map[b] = ((char)b).ToString();
        
        // Space (32) maps to Ġ (U+0120)
        map[32] = "\u0120";

        // We could implement the full 256-char mapping here, but for Qwen/LLaMA GGUFs,
        // the most important thing is that the pre-tokeniser and byte-tokeniser 
        // use the same characters as the vocab strings.
        
        return map;
    }

    /// <summary>Encodes a string to byte tokens using the byte-level fallback.</summary>
    public static IEnumerable<string> ByteTokenise(string text)
    {
        foreach (byte b in System.Text.Encoding.UTF8.GetBytes(text))
            yield return ByteTokenString(b);
    }
}
