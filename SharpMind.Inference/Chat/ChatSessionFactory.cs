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
            Transformer model, Tokenizer tokenizer, ModelMetaData? meta = null, IAgentBuilder? agentBuilder = null, IPromptPreProcessor? preProcessor = null, IPromptPostProcessor? postProcessor = null, IProgress<float>? progress = null, Func<ToolPermissionContext, Task<ToolPermission>>? permissions = null, int? seed = null)
        {
            var closedGen = generatorBuilderDef.IsGenericTypeDefinition
                ? generatorBuilderDef.MakeGenericType(cacheBuilder)
                : generatorBuilderDef;
            var sessionType = typeof(ChatSession<,>).MakeGenericType(closedGen, cacheBuilder);
            return (IChatSession)Activator.CreateInstance(sessionType, [model, tokenizer, meta, agentBuilder, preProcessor, postProcessor, progress, permissions, null, seed])!;
        }
        // Compile-time — for known type combos
        public static ChatSession<T, K> CreateChatSession<T, K>(
            Transformer model, Tokenizer tokenizer, ModelMetaData? meta = null, IAgentBuilder? agentBuilder = null, IPromptPreProcessor? preProcessor = null, IPromptPostProcessor? postProcessor = null, IProgress<float>? progress = null, Func<ToolPermissionContext, Task<ToolPermission>>? permissions = null, int? seed = null)
            where K : IKVCacheBuilder, new()
            where T : IGeneratorBuilder<K>, new()
            => new(model, tokenizer, meta, agentBuilder, preProcessor, postProcessor, progress, permissions, null, seed);
    }
}
