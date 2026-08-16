using SharpMind.Data.Sources;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.PreTokeniser;
using SharpMind.Tokenization.Serialisation;
using SharpMind.Tokenization.Vocab;

namespace SharpMind.Tokenization;

/// <summary>
/// High-level pipeline for training and persisting tokenizers.
/// </summary>
public static class TokenizationPipeline
{
    /// <summary>
    /// Trains a BPE model from a data source and saves it to disk.
    /// </summary>
    public static async Task<Tokenizer> TrainAndSaveAsync(
        IDataSource source, 
        string savePath, 
        int targetVocabSize = 32_000)
    {                
        var trainer = new BpeTrainer(targetVocabSize);

        var model = await trainer.TrainAsync(source.ReadAsync());
        TokenizerFile.Save(model, savePath);
        
        return new Tokenizer(model);
    }

    /// <summary>
    /// Builds a character-level tokenizer from a data source: the vocabulary is
    /// the sorted set of distinct characters in the corpus (plus the standard
    /// special tokens), so each token is one character — the classic GPT
    /// character-language-model setup. Unlike BPE there is nothing to train and
    /// nothing to cache; the vocab is derived from the corpus each call.
    /// </summary>
    public static async Task<Tokenizer> TrainCharacterAsync(IDataSource source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var chars = new HashSet<char>();
        await foreach (string doc in source.ReadAsync())
        {
            foreach (char c in doc)
                chars.Add(c);
        }

        // Characters first (sorted by code point, like Python's sorted(set(text))),
        // then the special tokens, so the data characters keep their natural IDs
        // and the reserved special slots never collide with corpus characters.
        var ordered = chars.OrderBy(c => c).Select(c => c.ToString()).ToList();
        var specials = new SpecialTokens();
        foreach (string special in specials.All)
        {
            if (!ordered.Contains(special))
                ordered.Add(special);
        }

        var vocab = new Vocabulary(ordered, specials);
        var model = new BpeModel(vocab, new List<MergeRule>(), new WhitespacePreTokeniser(), charMode: true);
        return new Tokenizer(model);
    }

    /// <summary>
    /// Loads a tokenizer from a saved file.
    /// </summary>
    public static Tokenizer Load(string path) => new(TokenizerFile.Load(path));
}
