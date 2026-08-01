using SharpMind.Tokenization.Vocab;
using SharpMind.Tokenization.Serialisation;
using SharpMind.Tokenization.Bpe;

namespace SharpMind.Tokenization;

/// <summary>
/// Top-level tokenizer. Wraps a trained <see cref="BpeModel"/> and exposes
/// encode, decode, and vocabulary access.
///
/// Construction:
/// <code>
/// // Train from a SharpMind.Data pipeline
/// var trainer   = new BpeTrainer(vocabSize: 32_000);
/// var model     = await trainer.TrainAsync(pipeline.ReadAsync());
/// var tokenizer = new Tokenizer(model);
/// TokenizerFile.Save(model, "tokenizer.json");
///
/// // Load SharpMind native
/// var tokenizer = Tokenizer.FromFile("tokenizer.json");
///
/// // Load from a specific model family
/// var tokenizer = Tokenizer.FromGpt2("encoder.json", "vocab.bpe");
/// var tokenizer = Tokenizer.FromLlama("tokenizer.json");
/// var tokenizer = Tokenizer.FromMistral("tokenizer.json");
///
/// // Load directly from GGUF vocab data (model-family agnostic)
/// var tokenizer = Tokenizer.FromGguf(tokens, merges, scores, tokenTypes, bosId, eosId);
/// </code>
///
/// Usage with SharpMind.Data:
/// <code>
/// var loader = new DataLoader(
///     pipeline,
///     tokenise: text => tokenizer.Encode(text),
///     batcher:  new PackingBatcher(
///                   batchSize:  8,
///                   maxSeqLen:  2048,
///                   eosTokenId: tokenizer.EosId,
///                   padTokenId: tokenizer.PadId));
/// </code>
/// </summary>
public class Tokenizer
{
    private readonly BpeModel _model;

    public Tokenizer(BpeModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
    }

    // Vocabulary

    public int VocabSize => _model.Vocab.Size;
    public int UnkId => _model.Vocab.UnkId;
    public int BosId => _model.Vocab.BosId;
    public int EosId => _model.Vocab.EosId;
    public int PadId => _model.Vocab.PadId;
    public Vocabulary Vocab => _model.Vocab;
    public SpecialTokens Specials => _model.Vocab.Specials;

    /// <summary>
    /// Returns all token IDs that should halt generation.
    /// Always includes <see cref="EosId"/>, plus any known turn-end tokens
    /// found in the vocabulary (e.g. &lt;|eot_id|&gt;, &lt;|im_end|&gt;,
    /// &lt;|end_of_text|&gt;, &lt;|end▁of▁sentence|&gt;, &lt;end_of_turn&gt;).
    /// </summary>
    public IReadOnlyList<int> GetEndOfGenerationIds()
    {
        var stopIds = new List<int> { EosId };
        string[] knownTurnEndTokens =
        [
            "<|eot_id|>",
            "<|im_end|>",
            "<|end_of_text|>",
            "<|end▁of▁sentence|>",
            "\uFFFD\uFFFDend\uFFFDof\uFFFDsentence\uFFFD\uFFFD",  // fallback encoding
            "<end_of_turn>",
            "</s>",
        ];
        foreach (string token in knownTurnEndTokens)
        {
            if (Vocab.Contains(token))
                stopIds.Add(Vocab.GetId(token));
        }
        // Deduplicate in case EosId matches one of the above
        return [.. stopIds.Distinct()];
    }

    // True for SentencePiece-style vocabularies (original LLaMA/LLaMA-2,
    // Mistral, TinyLlama, etc.) that ship no merges array. These rank
    // candidate merges by token score instead of an explicit rule list.
    public bool UseSentencePieceMerge => _model.Encoder.UseSentencePieceMerge;
    /// <summary>
    /// Encodes text to token IDs.
    /// Set <paramref name="addBos"/> / <paramref name="addEos"/> to include
    /// boundary tokens required by most autoregressive LLMs.
    /// </summary>
    public int[] Encode(string text, bool addBos = false, bool addEos = false) => _model.Encoder.Encode(text, addBos, addEos);

    /// <summary>Encodes a batch of strings independently.</summary>
    public int[][] EncodeBatch(IEnumerable<string> texts, bool addBos = false, bool addEos = false) => _model.Encoder.EncodeBatch(texts, addBos, addEos);

    /// <summary>Decodes token IDs back to a string. Skips special tokens by default.</summary>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecials = true) => _model.Encoder.Decode(ids, skipSpecials);

    public string Decode(int[] ids, bool skipSpecials = true) => _model.Encoder.Decode(ids, skipSpecials);

    public string IdToToken(int id) => _model.Vocab.GetToken(id);
    public int TokenToId(string tok) => _model.Vocab.GetId(tok);

    /// <summary>
    /// Adds an additional special token at runtime (e.g., a token that was
    /// missing from a GGUF's SentencePiece vocab but referenced in the chat
    /// template). Returns the assigned token ID, or the existing ID if the
    /// token was already in the vocabulary.
    /// </summary>
    public int AddAdditionalToken(string token)
    {
        bool isNew = !Vocab.Contains(token);
        int id = Vocab.AddToken(token);
        Vocab.Specials.AddAdditional(token);
        _model.Encoder.RefreshSpecials();
        return id;
    }

    /// <summary>Loads a SharpMind native tokenizer JSON file.</summary>
    public static Tokenizer FromFile(string path) => new(TokenizerFile.Load(path));

    /// <summary>Parses a SharpMind native tokenizer JSON string.</summary>
    public static Tokenizer FromJson(string json) => new(TokenizerFile.FromJson(json));

    /// <summary>Serialises this tokenizer to its SharpMind JSON string.</summary>
    public string ToJson() => TokenizerFile.ToJson(_model);

    /// <summary>Saves this tokenizer to a SharpMind native tokenizer JSON file.</summary>
    public void SaveJson(string path) => TokenizerFile.Save(_model, path);

    /// <summary>
    /// Builds a tokenizer directly from GGUF-embedded vocab data.
    ///
    /// This is the preferred loading path when a GGUF file is available:
    /// the vocab stored in GGUF is always byte-for-byte identical to what
    /// the model weights were trained against, so it can never produce the
    /// token-ID out-of-bounds crashes that occur when an external
    /// tokenizer.json has a mismatched vocab size.
    ///
    /// Supports both GPT-2 byte-level BPE (Qwen, Llama-3, Phi) and
    /// SentencePiece-based (LLaMA, LLaMA-2, Mistral, TinyLlama) models.
    /// The correct encoding path is selected automatically based on whether
    /// <c>tokenizer.ggml.merges</c> and <c>tokenizer.ggml.scores</c> are present.
    /// </summary>
    /// <param name="tokens">Vocab strings in ID order (tokenizer.ggml.tokens).</param>
    /// <param name="merges">Merge rules as "left right" strings (tokenizer.ggml.merges).</param>
    /// <param name="scores">Per-token scores (tokenizer.ggml.scores). May be null.</param>
    /// <param name="tokenTypes">Per-token type flags (tokenizer.ggml.token_type). May be null.</param>
    /// <param name="bosId">BOS token ID (tokenizer.ggml.bos_token_id).</param>
    /// <param name="eosId">EOS token ID (tokenizer.ggml.eos_token_id).</param>
    /// <param name="architecture">Model architecture string (general.architecture). Used to select the correct pre-tokeniser (cl100k vs GPT-2).</param>
    public static Tokenizer FromGguf(string[] tokens, string[]? merges, int[]? tokenTypes, int bosId, int eosId, float[]? scores = null, string? architecture = null) => new(GgufConverter.Convert(tokens, merges, tokenTypes, bosId, eosId, scores, architecture));

    /// <summary>
    /// Loads a GPT-2 tokenizer from its two native files.
    /// <paramref name="encoderJsonPath"/> = <c>encoder.json</c>
    /// <paramref name="vocabBpePath"/>    = <c>vocab.bpe</c>
    /// </summary>
    public static Tokenizer FromGpt2(string encoderJsonPath, string vocabBpePath) => new(Gpt2Converter.Convert(encoderJsonPath, vocabBpePath));

    /// <summary>
    /// Loads a LLaMA 2 or LLaMA 3 tokenizer from a HuggingFace
    /// <c>tokenizer.json</c> file.
    /// </summary>
    public static Tokenizer FromLlama(string tokenizerJsonPath) => new(LlamaConverter.Convert(tokenizerJsonPath));

    /// <summary>
    /// Loads a Mistral tokenizer from a HuggingFace <c>tokenizer.json</c> file.
    /// Handles both v0.1 (LLaMA-compatible) and v0.3+ (extended vocab) formats.
    /// </summary>
    public static Tokenizer FromMistral(string tokenizerJsonPath) => new(MistralConverter.Convert(tokenizerJsonPath));

    /// <summary>
    /// Loads a Qwen tokenizer from a HuggingFace <c>tokenizer.json</c> file.
    /// Handles Qwen-specific special tokens like <|im_start|>, <|im_end|>.
    /// </summary>
    public static Tokenizer FromQwen(string tokenizerJsonPath) => new(QwenConverter.Convert(tokenizerJsonPath));
}
