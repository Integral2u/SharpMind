using SharpMind.Data.Sources;
using SharpMind.Tokenization.Bpe;
using SharpMind.Tokenization.Serialisation;

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
    /// Loads a tokenizer from a saved file.
    /// </summary>
    public static Tokenizer Load(string path)
    {
        var model = TokenizerFile.Load(path);
        return new Tokenizer(model);
    }
}
