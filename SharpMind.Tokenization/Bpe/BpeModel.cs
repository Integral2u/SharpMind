using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Serialisation;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tokenization.Bpe;

/// <summary>
/// The result of BPE training — a fully trained model ready for encoding.
/// Holds the vocabulary, ordered merge rules, and a pre-configured encoder.
/// Passed to <see cref="TokenizerFile.Save"/> or directly to
/// <see cref="Tokenization"/>.
/// </summary>
public sealed class BpeModel
{
    internal BpeModel(
        Vocabulary vocab,
        List<MergeRule> merges,
        IPreTokeniser preTokeniser,
        IReadOnlyList<float>? tokenScores = null,
        bool charMode = false)
    {
        Vocab = vocab;
        Merges = merges.AsReadOnly();
        PreTokeniser = preTokeniser;
        IsCharMode = charMode;
        Encoder = new BpeEncoder(vocab, Merges, preTokeniser, tokenScores, charMode);
    }

    public Vocabulary Vocab { get; }
    public IReadOnlyList<MergeRule> Merges { get; }
    public IPreTokeniser PreTokeniser { get; }
    public BpeEncoder Encoder { get; }

    /// <summary>
    /// True when this is a character-level tokenizer (each token is one corpus
    /// character) rather than a byte-pair-encoding model. Encodes via a direct
    /// character→id map; there are no merge rules in char mode.
    /// </summary>
    public bool IsCharMode { get; }
}
