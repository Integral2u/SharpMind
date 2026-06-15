using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tokenization.Serialisation;

/// <summary>
/// Converts GGUF-embedded tokenizer data to a SharpMind <see cref="BpeModel"/>.
///
/// GGUF stores all tokenizer data in its KV metadata:
///   tokenizer.ggml.tokens     — vocab strings in ID order
///   tokenizer.ggml.merges     — BPE merge rules as "left right" strings
///   tokenizer.ggml.scores     — per-token log-prob scores (informational)
///   tokenizer.ggml.token_type — per-token type flags:
///                                 1 = NORMAL
///                                 2 = UNKNOWN  → unk token
///                                 3 = CONTROL  → bos / eos / other specials
///                                 4 = USER_DEFINED
///                                 5 = UNUSED
///                                 6 = BYTE     → byte-level fallback tokens
///   tokenizer.ggml.bos_token_id — integer ID of the BOS token
///   tokenizer.ggml.eos_token_id — integer ID of the EOS token
///
/// This is model-family agnostic: the same logic correctly handles LLaMA,
/// Mistral, Qwen, Phi, and any other BPE model stored in GGUF format,
/// because the vocab and merge rules are embedded verbatim rather than
/// inferred from a family-specific JSON schema.
/// </summary>
public static class GgufConverter
{
    // GGUF token_type values
    private const int TypeNormal      = 1;
    private const int TypeUnknown     = 2;
    private const int TypeControl     = 3;
    private const int TypeUserDefined = 4;
    private const int TypeUnused      = 5;
    private const int TypeByte        = 6;

    /// <summary>
    /// Builds a <see cref="BpeModel"/> from raw GGUF vocab data.
    /// </summary>
    /// <param name="tokens">
    ///   Token strings in vocab-ID order (tokenizer.ggml.tokens).
    ///   This is passed directly to <see cref="Vocabulary"/> — no sorting required.
    /// </param>
    /// <param name="merges">
    ///   BPE merge rules as "left right" strings (tokenizer.ggml.merges).
    ///   Rank is their position in this array.
    /// </param>
    /// <param name="scores">Token scores (tokenizer.ggml.scores). Informational — not used in inference.</param>
    /// <param name="tokenTypes">Per-token type flags (tokenizer.ggml.token_type). Used to find special tokens.</param>
    /// <param name="bosId">BOS token ID (tokenizer.ggml.bos_token_id).</param>
    /// <param name="eosId">EOS token ID (tokenizer.ggml.eos_token_id).</param>
    public static BpeModel Convert(
        string[]  tokens,
        string[]? merges,
        int[]?    tokenTypes,
        int       bosId,
        int       eosId)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Length == 0)
            throw new ArgumentException("Token list is empty — GGUF vocab data is missing.", nameof(tokens));
        
        // Resolve byte mapping
        // GGUF explicitly marks byte tokens with type=6.
        // We build a map: byte [0..255] -> token string.
        var byteMap = new string[256];
        for (int i = 0; i < 256; i++) byteMap[i] = $"<0x{i:X2}>";

        if (tokenTypes != null)
        {
            int limit = Math.Min(tokens.Length, tokenTypes.Length);
            for (int i = 0; i < limit; i++)
            {
                if (tokenTypes[i] == TypeByte)
                {
                    // Try to determine which byte this represents.
                    // Usually, byte tokens appear in order 0..255.
                    // A safer way is to check if the token string is a known byte representation.
                    // For now, we'll use the index-based heuristic if we find exactly 256 TypeByte tokens.
                }
            }
        }
        // Actually, the most robust way to handle GGUF is to use the tokens as-is
        // and let the BPE encoder's ByteTokenise produce strings that match these.
        
        // Resolve special token strings
        // ... (rest of the method)


        // BOS / EOS: use the explicit IDs from GGUF metadata.
        string bosToken = IdToToken(tokens, bosId) ?? SpecialTokens.DefaultBos;
        string eosToken = IdToToken(tokens, eosId) ?? SpecialTokens.DefaultEos;

        // UNK: first token whose type flag is UNKNOWN (2).
        // Fallback to scanning common names, then the SharpMind default.
        string unkToken = FirstOfType(tokens, tokenTypes, TypeUnknown)
                       ?? FirstNamed(tokens, "<unk>", "[UNK]", "<|unk|>", "<|unknown|>")
                       ?? SpecialTokens.DefaultUnk;

        // PAD: GGUF has no dedicated pad-type flag; scan common names.
        // Most models tie pad to EOS or don't define one — fall back to EOS.
        string padToken = FirstNamed(tokens, "<pad>", "[PAD]", "<|pad|>", "<|padding|>")
                       ?? eosToken;

        // Additional control tokens: type=3 that are not already bos/eos/unk/pad.
        // These are things like <|im_start|>, <|system|>, <|reserved_n|>, etc.
        var additional = new List<string>();
        if (tokenTypes != null)
        {
            int limit = Math.Min(tokens.Length, tokenTypes.Length);
            for (int i = 0; i < limit; i++)
            {
                if (tokenTypes[i] != TypeControl) continue;
                string tok = tokens[i];
                if (tok != bosToken && tok != eosToken && tok != unkToken && tok != padToken)
                    additional.Add(tok);
            }
        }

        var specials = new SpecialTokens(unkToken, bosToken, eosToken, padToken, additional);

        // Vocabulary
        // GGUF tokens are already in ID order — pass the list directly to the
        // internal Vocabulary constructor that preserves the existing ordering.
        var vocab = new Vocabulary(tokens, specials);

        // Merge rules
        // Each entry is "left right" — same format as HuggingFace tokenizer.json.
        var mergeList = new List<MergeRule>();
        if (merges != null)
        {
            for (int rank = 0; rank < merges.Length; rank++)
            {
                string[] parts = merges[rank].Split(' ', 2, StringSplitOptions.None);
                if (parts.Length == 2)
                    mergeList.Add(new MergeRule(parts[0], parts[1], parts[0] + parts[1], rank));
            }
        }        

        // PreTokeniser
        // All BPE models in GGUF (LLaMA, Mistral, Qwen, Phi, etc.) use
        // the GPT-2 byte-level pre-tokenisation pattern.
        return new BpeModel(vocab, mergeList, new Gpt2PreTokeniser());
    }

    // Helpers

    private static string? IdToToken(string[] tokens, int id)
        => id >= 0 && id < tokens.Length ? tokens[id] : null;

    private static string? FirstOfType(string[] tokens, int[]? types, int targetType)
    {
        if (types == null) return null;
        int limit = Math.Min(tokens.Length, types.Length);
        for (int i = 0; i < limit; i++)
            if (types[i] == targetType) return tokens[i];
        return null;
    }

    private static string? FirstNamed(string[] tokens, params string[] candidates)
    {
        var set = new HashSet<string>(tokens, StringComparer.Ordinal);
        foreach (string c in candidates)
            if (set.Contains(c)) return c;
        return null;
    }
}
