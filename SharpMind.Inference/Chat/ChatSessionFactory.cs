using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat.PromptFormatters;
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
            Transformer model, Tokenizer tokenizer, ModelMetaData? meta = null, IAgentBuilder? agentBuilder = null, IPromptPreProcessor? preProcessor = null, IPromptPostProcessor? postProcessor = null, IProgress<float>? progress = null, Func<ToolPermissionContext, Task<ToolPermission>>? permissions = null, IChatPromptFormatter? formatter = null, int? seed = null, bool disposeModel = false, int? maxCacheLen = null, IInferenceEngine? engine = null, object? generatorFactory = null, InterceptingFileSystem? fileSystem = null, InterceptingNetworkHandler? networkHandler = null)
        {
            var closedGen = generatorBuilderDef.IsGenericTypeDefinition
                ? generatorBuilderDef.MakeGenericType(cacheBuilder)
                : generatorBuilderDef;
            var sessionType = typeof(ChatSession<,>).MakeGenericType(closedGen, cacheBuilder);
            // disposeModel defaults false: the caller loaded the model and keeps it
            // across sessions. This used to be hardcoded true, so ending one chat
            // disposed the shared Transformer and every later session threw
            // ObjectDisposedException — swallowed and shown as an empty reply.
            //
            // generatorFactory is the closed Func<IGenerator<K>> (batched serving):
            // the Activator binds it to the session's parameter of that exact type.
            return (IChatSession)Activator.CreateInstance(sessionType, [model, tokenizer, meta, agentBuilder, preProcessor, postProcessor, progress, permissions, null, formatter, seed, disposeModel, maxCacheLen, engine, null, generatorFactory, fileSystem, networkHandler])!;
        }
        // Compile-time — for known type combos
        public static ChatSession<T, K> CreateChatSession<T, K>(
            Transformer model, Tokenizer tokenizer, ModelMetaData? meta = null, IAgentBuilder? agentBuilder = null, IPromptPreProcessor? preProcessor = null, IPromptPostProcessor? postProcessor = null, IProgress<float>? progress = null, Func<ToolPermissionContext, Task<ToolPermission>>? permissions = null, IChatPromptFormatter? formatter = null, int? seed = null, bool disposeModel = false, int? maxCacheLen = null, IInferenceEngine? engine = null, Func<IGenerator<K>>? generatorFactory = null, InterceptingFileSystem? fileSystem = null, InterceptingNetworkHandler? networkHandler = null)
            where K : IKVCacheBuilder, new()
            where T : IGeneratorBuilder<K>, new()
            => new(model, tokenizer, meta, agentBuilder, preProcessor, postProcessor, progress, permissions, null, formatter, seed, disposeModel, maxCacheLen, engine, null, generatorFactory, fileSystem, networkHandler);
    }
}
