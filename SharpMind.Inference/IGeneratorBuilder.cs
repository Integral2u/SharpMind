using SharpMind.Model;

namespace SharpMind.Inference
{
    public interface IGeneratorBuilder<T> where T : IKVCacheBuilder, new()
    {
        public IGenerator<T> CreateGenerator(Transformer model, Tokenization.Tokenizer tokenizer, bool addBos, bool addEos, IKVCache[]? caches, int? seed = null);
    }

    public class StandardGeneratorBuilder<T> : IGeneratorBuilder<T> where T : IKVCacheBuilder, new()
    {
        public IGenerator<T> CreateGenerator(Transformer model, Tokenization.Tokenizer tokenizer, bool addBos, bool addEos, IKVCache[]? caches, int? seed = null) => new StandardGenerator<T>(model, tokenizer, addBos, addEos, caches, seed);
    }
    public class SpeculativeGeneratorBuilder<T> : IGeneratorBuilder<T> where T : IKVCacheBuilder, new()
    {
        public IGenerator<T> CreateGenerator(Transformer model, Tokenization.Tokenizer tokenizer, bool addBos, bool addEos, IKVCache[]? caches, int? seed = null) => new SpeculativeGenerator<T>(model, tokenizer, addBos, addEos, caches, seed);
    }
}
