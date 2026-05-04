using SharpMind.Tokenizer.Vocab;
using SharpMind.Tokenizer.Serialisation;
using SharpMind.Tokenizer.Bpe;

namespace SharpMind.Tokenizer;

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
public sealed class Tokenization
{
    private readonly BpeModel _model;

    public Tokenization(BpeModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
    }

    // ── Vocabulary ────────────────────────────────────────────────────────

    public int VocabSize => _model.Vocab.Size;
    public int UnkId => _model.Vocab.UnkId;
    public int BosId => _model.Vocab.BosId;
    public int EosId => _model.Vocab.EosId;
    public int PadId => _model.Vocab.PadId;
    public Vocabulary Vocab => _model.Vocab;
    public SpecialTokens Specials => _model.Vocab.Specials;

    // ── Encode ────────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes text to token IDs.
    /// Set <paramref name="addBos"/> / <paramref name="addEos"/> to include
    /// boundary tokens required by most autoregressive LLMs.
    /// </summary>
    public int[] Encode(string text, bool addBos = false, bool addEos = false)
        => _model.Encoder.Encode(text, addBos, addEos);

    /// <summary>Encodes a batch of strings independently.</summary>
    public int[][] EncodeBatch(IEnumerable<string> texts, bool addBos = false, bool addEos = false)
        => _model.Encoder.EncodeBatch(texts, addBos, addEos);

    // ── Decode ────────────────────────────────────────────────────────────

    /// <summary>Decodes token IDs back to a string. Skips special tokens by default.</summary>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecials = true)
        => _model.Encoder.Decode(ids, skipSpecials);

    public string Decode(int[] ids, bool skipSpecials = true)
        => _model.Encoder.Decode(ids, skipSpecials);

    // ── Token helpers ─────────────────────────────────────────────────────

    public string IdToToken(int id) => _model.Vocab.GetToken(id);
    public int TokenToId(string tok) => _model.Vocab.GetId(tok);

    // ── Factory: SharpMind native ─────────────────────────────────────────

    /// <summary>Loads a SharpMind native tokenizer JSON file.</summary>
    public static Tokenization FromFile(string path)
        => new(TokenizerFile.Load(path));

    // ── Factory: model-family converters ──────────────────────────────────

    /// <summary>
    /// Loads a GPT-2 tokenizer from its two native files.
    /// <paramref name="encoderJsonPath"/> = <c>encoder.json</c>
    /// <paramref name="vocabBpePath"/>    = <c>vocab.bpe</c>
    /// </summary>
    public static Tokenization FromGpt2(string encoderJsonPath, string vocabBpePath)
        => new(Gpt2Converter.Convert(encoderJsonPath, vocabBpePath));

    /// <summary>
    /// Loads a LLaMA 2 or LLaMA 3 tokenizer from a HuggingFace
    /// <c>tokenizer.json</c> file.
    /// </summary>
    public static Tokenization FromLlama(string tokenizerJsonPath)
        => new(LlamaConverter.Convert(tokenizerJsonPath));

    /// <summary>
    /// Loads a Mistral tokenizer from a HuggingFace <c>tokenizer.json</c> file.
    /// Handles both v0.1 (LLaMA-compatible) and v0.3+ (extended vocab) formats.
    /// </summary>
    public static Tokenization FromMistral(string tokenizerJsonPath)
        => new(MistralConverter.Convert(tokenizerJsonPath));
}
