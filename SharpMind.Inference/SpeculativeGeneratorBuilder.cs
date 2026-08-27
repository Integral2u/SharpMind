using SharpMind.Model;

namespace SharpMind.Inference
{
    public class SpeculativeGeneratorBuilder<T> : IGeneratorBuilder<T> where T : IKVCacheBuilder, new()
    {
        public IGenerator<T> CreateGenerator(Transformer model, Tokenization.Tokenizer tokenizer, bool addBos, bool addEos, IKVCache[]? caches, int? seed = null, int? maxCacheLen = null) => new SpeculativeGenerator<T>(model, tokenizer, addBos, addEos, caches, seed, maxCacheLen);
    }
}
