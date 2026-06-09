using SharpMind.Inference.Agent;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat
{
    public static class ChatSessionFactory
    {
        // Reflection-based — for your iterator loop (runtime type selection)
        public static IChatSession CreateChatSession(
            Type generatorBuilderDef,  // typeof(StandardGeneratorBuilder<>)
            Type cacheBuilder,         // typeof(KVCacherBuilder)
            Transformer model, Tokenizer tokenizer, ModelMetaData? meta = null, IAgentBuilder? agentBuilder = null, int? seed = null, InterceptingFileSystem? fileSystem = null, InterceptingNetworkHandler? networkHandler = null)
        {
            var closedGen = generatorBuilderDef.MakeGenericType(cacheBuilder);
            var sessionType = typeof(ChatSession<,>).MakeGenericType(closedGen, cacheBuilder);
            return (IChatSession)Activator.CreateInstance(sessionType, [model, tokenizer, meta, agentBuilder, null, null, seed])!;
        }
        // Compile-time — for known type combos
        public static ChatSession<T, K> CreateChatSession<T, K>(
            Transformer model, Tokenizer tokenizer, ModelMetaData? meta = null, IAgentBuilder? agentBuilder = null, int? seed = null, InterceptingFileSystem? fileSystem = null, InterceptingNetworkHandler? networkHandler = null)
            where K : IKVCacheBuilder, new()
            where T : IGeneratorBuilder<K>, new()
            => new(model, tokenizer, meta, agentBuilder, null, null, seed);
    }
}
