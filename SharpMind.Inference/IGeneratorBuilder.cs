using SharpMind.Model;

namespace SharpMind.Inference
{
    public interface IGeneratorBuilder<T> where T : IKVCacheBuilder, new()
    {
        public IGenerator<T> CreateGenerator(Transformer model, Tokenization.Tokenizer tokenizer, bool addBos, bool addEos, IKVCache[]? caches, int? seed = null);
    }
}
