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
/// Handles both GPT-2 byte-level BPE (Qwen, Llama-3/tiktoken, Phi) and
/// SentencePiece-based (LLaMA, LLaMA-2, Mistral, TinyLlama) models.
/// The correct encoding path is selected automatically based on whether
/// <c>tokenizer.ggml.merges</c> and <c>tokenizer.ggml.scores</c> are present.
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
    /// <param name="scores">
    ///   Token scores (tokenizer.ggml.scores). Used to rank candidate merges
    ///   when <paramref name="merges"/> is null/empty — SentencePiece-style
    ///   GGUFs (LLaMA/LLaMA-2, Mistral, TinyLlama) have no explicit merges
    ///   array, so <see cref="Bpe.BpeEncoder"/> falls back to score-ranked
    ///   greedy merging in that case.
    /// </param>
    /// <param name="tokenTypes">Per-token type flags (tokenizer.ggml.token_type). Used to find special tokens.</param>
    /// <param name="bosId">BOS token ID (tokenizer.ggml.bos_token_id).</param>
    /// <param name="eosId">EOS token ID (tokenizer.ggml.eos_token_id).</param>
    public static BpeModel Convert(
        string[]  tokens,
        string[]? merges,
        int[]?    tokenTypes,
        int       bosId,
        int       eosId,
        float[]?  scores = null,
        string?   architecture = null)
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
        // Tokens are passed through as-is; the BPE encoder handles byte-level
        // fallback during encode. No byte-map preprocessing is needed.
        
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
        // tiktoken-based models (Llama 3+, Qwen 2+/3, DeepSeek, Phi-3/4,
        // Gemma 2, GPT-4o) use cl100k-style pre-tokenisation with case-
        // insensitive contraction matching. All other byte-level BPE models
        // (GPT-2, GPT-Neo, Bloom, Starcoder, Phi-1/2, etc.) use the original
        // GPT-2 pattern with case-sensitive contractions.
        // Original LLaMA/LLaMA-2/Mistral/TinyLlama GGUFs are SentencePiece-
        // based: they have no merges array at all, and instead rank merges by
        // tokenizer.ggml.scores. mergeList will be empty in that case, and
        // BpeEncoder detects it (via the scores array below) and switches to
        // score-ranked SentencePiece-style merging instead of silently
        // tokenising everything byte-by-byte.
        IPreTokeniser preTokeniser = SelectPreTokeniser(architecture, merges);
        return new BpeModel(vocab, mergeList, preTokeniser, scores);
    }

    /// <summary>
    /// Selects the correct <see cref="IPreTokeniser"/> based on the model
    /// architecture string from GGUF metadata (<c>general.architecture</c>).
    ///
    /// tiktoken-based models (Llama 3+, Qwen, DeepSeek, Phi-3/4, Gemma 2, DBRX)
    /// use cl100k-style pattern with case-insensitive contraction matching.
    /// Everything else uses the classic GPT-2 pattern.
    /// </summary>
    private static IPreTokeniser SelectPreTokeniser(string? architecture, string[]? merges = null)
    {
        if (string.IsNullOrWhiteSpace(architecture))
            return new Gpt2PreTokeniser();

        string arch = architecture.ToLowerInvariant();

        // Llama 1/2 uses SentencePiece (no merges array); Llama 3+ uses tiktoken (has merges)
        bool hasMerges = merges is { Length: > 0 };

        // Architectures known to use tiktoken-based vocabularies
        if ((arch.StartsWith("llama") && hasMerges)  // Llama 3+ (tiktoken)
            || arch is "qwen2" or "qwen2.5" or "qwen3" or "qwen2moe" or "deepseek2"
            || arch.Contains("qwen")
            || arch.Contains("deepseek")
            || arch.StartsWith("phi3") || arch.StartsWith("phi4")
            || arch.StartsWith("gemma2") || arch.StartsWith("gemma3") || arch.StartsWith("gemma-3")
            || arch is "dbrx"
            || arch is "mistral3" or "ministral")
            return new Cl100kPreTokeniser();

        return new Gpt2PreTokeniser();
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
