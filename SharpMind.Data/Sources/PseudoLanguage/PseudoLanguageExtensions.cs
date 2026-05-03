using SharpMind.Data.Batching;

namespace SharpMind.Data.Sources.PseudoLanguage;

public static class PseudoLanguageExtensions
{
    public static DataLoader ToDataLoader(
        this PseudoLanguagePipeline pipeline,
        int batchSize,
        int maxSeqLen,
        int eosTokenId = 2,
        int padTokenId = 0)
    {
        var generator = pipeline.Generator;

        int[] Tokenize(string text)
        {
            return [.. text.Split(' ', StringSplitOptions.RemoveEmptyEntries)
                .Select(word => generator.TextToId(word))
                .Where(id => id >= 0)];
        }

        var batcher = new PackingBatcher(batchSize, maxSeqLen, eosTokenId, padTokenId);
        return new DataLoader(pipeline, Tokenize, batcher);
    }
}