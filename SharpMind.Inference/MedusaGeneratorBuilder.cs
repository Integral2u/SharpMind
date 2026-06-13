using SharpMind.Model;
using SharpMind.Model.Layers;

namespace SharpMind.Inference;

public class MedusaGeneratorBuilder<T> : IGeneratorBuilder<T> where T : IKVCacheBuilder, new()
{
    private const int DefaultNumHeads = 3;

    public IGenerator<T> CreateGenerator(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        bool addBos, bool addEos,
        IKVCache[]? caches,
        int? seed = null)
    {
        var lmHeadWeight = model.LmHead ?? model.EmbeddingWeight;
        var medusaHeads = new MedusaHeads(
            numHeads: DefaultNumHeads,
            hiddenDim: model.Config.HiddenDim,
            vocabSize: model.Config.VocabSize,
            lmHeadWeight: lmHeadWeight,
            rawEmbedding: null,
            rawDtype: null,
            qOps: null);

        return new MedusaGenerator<T>(
            model, tokenizer, addBos, addEos,
            medusaHeads, caches, seed);
    }
}
