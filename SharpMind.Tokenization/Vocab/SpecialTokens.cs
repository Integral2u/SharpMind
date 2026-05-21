namespace SharpMind.Tokenization.Vocab;

/// <summary>
/// Holds the special token strings and their IDs.
/// Special tokens are always added to the vocabulary before any BPE merges
/// so their IDs are small and stable.
///
/// Defaults match the LLaMA / SentencePiece convention:
///   0 = [UNK]   1 = [BOS]   2 = [EOS]   3 = [PAD]
/// Override via <see cref="SpecialTokensConfig"/> when constructing.
/// </summary>
public sealed class SpecialTokens
{
    // ── Default token strings ─────────────────────────────────────────────

    public const string DefaultUnk = "[UNK]";
    public const string DefaultBos = "[BOS]";
    public const string DefaultEos = "[EOS]";
    public const string DefaultPad = "[PAD]";

    // ── Resolved token strings ────────────────────────────────────────────

    public string Unk { get; }
    public string Bos { get; }
    public string Eos { get; }
    public string Pad { get; }

    /// <summary>Additional user-defined special tokens in insertion order.</summary>
    public IReadOnlyList<string> Additional { get; private set; }

    /// <summary>All special tokens in the order they are assigned IDs.</summary>
    public IReadOnlyList<string> All => [Unk, Bos, Eos, Pad, .. Additional];

    public SpecialTokens(SpecialTokensConfig? config = null)
    {
        Unk = config?.Unk ?? DefaultUnk;
        Bos = config?.Bos ?? DefaultBos;
        Eos = config?.Eos ?? DefaultEos;
        Pad = config?.Pad ?? DefaultPad;
        Additional = config?.Additional is List<string> list ? list : [.. (config?.Additional ?? [])];
    }

    /// <summary>
    /// Constructs from explicit token strings — used when loading from
    /// a saved vocabulary where IDs must match exactly.
    /// </summary>
    internal SpecialTokens(string unk, string bos, string eos, string pad,
                           IReadOnlyList<string> additional)
    {
        Unk = unk;
        Bos = bos;
        Eos = eos;
        Pad = pad;
        Additional = additional is List<string> list ? list : [.. additional];
    }

    /// <summary>Adds an additional special token for runtime injection.</summary>
    internal void AddAdditional(string token)
    {
        if (Additional is List<string> mutable)
        {
            if (!mutable.Contains(token))
                mutable.Add(token);
        }
    }
}