using SharpMind.Model.Tokenizer.PreTokeniser;
using SharpMind.Model.Tokenizer.Vocab;

namespace SharpMind.Model.Tokenizer.Bpe;

/// <summary>
/// The result of BPE training — a fully trained model ready for encoding.
/// Holds the vocabulary, ordered merge rules, and a pre-configured encoder.
/// Passed to <see cref="Serialisation.TokenizerFile.Save"/> or directly to
/// <see cref="Tokenizer"/>.
/// </summary>
public sealed class BpeModel
{
    internal BpeModel(
        Vocabulary vocab,
        List<MergeRule> merges,
        IPreTokeniser preTokeniser)
    {
        Vocab = vocab;
        Merges = merges.AsReadOnly();
        PreTokeniser = preTokeniser;
        Encoder = new BpeEncoder(vocab, Merges, preTokeniser);
    }

    public Vocabulary Vocab { get; }
    public IReadOnlyList<MergeRule> Merges { get; }
    public IPreTokeniser PreTokeniser { get; }
    public BpeEncoder Encoder { get; }
}
