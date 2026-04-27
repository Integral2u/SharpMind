using SharpMind.Model.Tokenizer.Serialisation;
using SharpMind.Model.Tokenizer.Vocab;
using SharpMind.Model.Tokenizer.Bpe;
using SharpMind.Model.Tokenizer.PreTokeniser;

namespace SharpMind.Model.Tokenizer;

/// <summary>
/// Top-level tokenizer API. Wraps a trained <see cref="BpeModel"/> and
/// exposes encode, decode, and vocabulary access in one place.
///
/// Construction:
/// <code>
/// // Train from scratch
/// var trainer   = new BpeTrainer(vocabSize: 32_000);
/// var model     = trainer.Train(documents);
/// var tokenizer = new Tokenizer(model);
/// TokenizerFile.Save(model, "tokenizer.json");
///
/// // Load SharpMind native
/// var tokenizer = Tokenizer.FromFile("tokenizer.json");
///
/// // Load HuggingFace tokenizer (GPT-2, LLaMA, Mistral…)
/// var tokenizer = Tokenizer.FromHuggingFace("path/to/tokenizer.json");
/// </code>
///
/// Usage with SharpMind.Data:
/// <code>
/// var loader = new DataLoader(
///     pipeline,
///     tokenise: text => tokenizer.Encode(text),
///     batcher:  new PackingBatcher(batchSize: 8, maxSeqLen: 2048,
///                                  eosTokenId: tokenizer.EosId,
///                                  padTokenId: tokenizer.PadId));
/// </code>
/// </summary>
public sealed class Tokenizer
{
    private readonly BpeModel _model;

    public Tokenizer(BpeModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
    }

    // ── Vocabulary properties ─────────────────────────────────────────────

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
    /// Set <paramref name="addBos"/> / <paramref name="addEos"/> to prepend/append
    /// the boundary tokens required by most autoregressive LLMs.
    /// </summary>
    public int[] Encode(string text, bool addBos = false, bool addEos = false)
        => _model.Encoder.Encode(text, addBos, addEos);

    /// <summary>Encodes a batch of strings independently.</summary>
    public int[][] EncodeBatch(IEnumerable<string> texts, bool addBos = false, bool addEos = false)
        => _model.Encoder.EncodeBatch(texts, addBos, addEos);

    // ── Decode ────────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes token IDs back to a string.
    /// Special tokens are skipped by default.
    /// </summary>
    public string Decode(ReadOnlySpan<int> ids, bool skipSpecials = true)
        => _model.Encoder.Decode(ids, skipSpecials);

    /// <summary>Decodes token IDs from an array.</summary>
    public string Decode(int[] ids, bool skipSpecials = true)
        => _model.Encoder.Decode(ids, skipSpecials);

    // ── Single-token helpers ──────────────────────────────────────────────

    /// <summary>Returns the token string for a given ID.</summary>
    public string IdToToken(int id) => _model.Vocab.GetToken(id);

    /// <summary>Returns the ID for a given token string, or UnkId if not found.</summary>
    public int TokenToId(string token) => _model.Vocab.GetId(token);

    // ── Factory methods ───────────────────────────────────────────────────

    /// <summary>Loads a SharpMind native tokenizer JSON file.</summary>
    public static Tokenizer FromFile(string path)
        => new(TokenizerFile.Load(path));

    /// <summary>
    /// Loads a HuggingFace <c>tokenizer.json</c> file.
    /// Compatible with GPT-2, LLaMA 2/3, Mistral, Falcon, and any HF BPE tokenizer.
    /// </summary>
    public static Tokenizer FromHuggingFace(string path)
        => new(TokenizerFile.LoadHuggingFace(path));
}
