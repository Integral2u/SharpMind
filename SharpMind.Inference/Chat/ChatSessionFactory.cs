using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat;

public static class ChatSessionFactory
{
    public static ChatSession<StandardGenerator> CreateStandard(
        Transformer model,
        Tokenizer tokenizer,
        GgufMeta? meta = null,
        IKVCache[]? caches = null,
        int? seed = null)
    {
        var generator = new StandardGenerator(model, tokenizer, caches, seed);
        return new ChatSession<StandardGenerator>(generator, tokenizer, meta);
    }

    public static ChatSession<SpeculativeGenerator> CreateSpeculative(
        Transformer model,
        Tokenizer tokenizer,
        GgufMeta? meta = null,
        IKVCache[]? caches = null,
        int? seed = null)
    {
        var generator = new SpeculativeGenerator(model, tokenizer, caches, seed);
        return new ChatSession<SpeculativeGenerator>(generator, tokenizer, meta);
    }
}
