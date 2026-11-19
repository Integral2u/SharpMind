using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Server.Protocol;

namespace SharpMind.Server;

/// <summary>
/// Creates IChatSession instances from a loaded model and an OpenAI chat
/// completion request. Each request gets its own session (the KV-cache is
/// rebuilt from the conversation history).
/// </summary>
public sealed class SessionFactory(SharpMindServerOptions options, PrefixCache prefixCache, BatchService? batches = null)
{
    private readonly SharpMindServerOptions _options = options;
    private readonly PrefixCache _prefixCache = prefixCache;
    private readonly BatchService? _batches = batches;

    /// <summary>
    /// Build the permission callback from current options. Returns null when
    /// no IO restrictions are active (no file or network gating needed).
    /// </summary>
    private Func<ToolPermissionContext, Task<ToolPermission>>? BuildPermissionCallback()
    {
        bool gateFile = _options.DisableFileIO;
        bool gateNetwork = _options.DisableNetworkIO;
        if (!gateFile && !gateNetwork) return null;

        return async ctx =>
        {
            await Task.CompletedTask;
            if (gateFile && ctx.Category == ToolCategory.File)
                return ToolPermission.Never;
            if (gateNetwork && ctx.Category == ToolCategory.Network)
                return ToolPermission.Never;
            return ToolPermission.Always;
        };
    }

    /// <summary>
    /// Create a fresh session, replay the conversation history, apply request
    /// parameters, and return the ready-to-use session. The last user message
    /// is returned separately - it must be passed to
    /// <see cref="IChatSession.GetResponseStreamAsync"/> to generate a response.
    /// </summary>
    public async Task<(IChatSession session, string lastUserMessage, bool continueFromHistory)> CreateSessionAsync(
        LoadedModel loaded, CreateChatCompletionRequest request)
    {
        // OpenAI tools are executed by the API client, not the server. Register
        // them so the model sees their schema, then hand every call back.
        var agentBuilder = OpenAiMapper.CreateToolAgent(request.Tools);

        // Batched chat: the session's generators enqueue into the model's shared
        // batcher. No per-session cache exists there, so the prefix cache is
        // bypassed below (a restored snapshot would land in the batcher's rows).
        Func<IGenerator<KVCacherBuilder>>? generatorFactory = null;
        BatchedGenerator? batcher = null;
        if (_options.BatchChatCompletions && _batches is not null)
        {
            batcher = _batches.Get(loaded);
            bool addBos = ModelMetaData.ResolveAddBos(loaded.Meta, loaded.Tokenizer.UseSentencePieceMerge);
            bool addEos = ModelMetaData.ResolveAddEos(loaded.Meta);
            int layers = loaded.Model.Config.NumLayers;
            generatorFactory = () => new BatchedGeneratorAdapter<KVCacherBuilder>(
                batcher, loaded.Tokenizer, layers, addBos, addEos);
        }

        // Build the session via the standard factory
        var session = ChatSessionFactory.CreateChatSession(
            typeof(StandardGeneratorBuilder<KVCacherBuilder>),
            typeof(KVCacherBuilder),
            loaded.Model,
            loaded.Tokenizer,
            loaded.Meta,
            agentBuilder: agentBuilder,
            preProcessor: null,
            postProcessor: null,
            progress: null,
            permissions: BuildPermissionCallback(),
            formatter: null,
            seed: null,
            maxCacheLen: _options.MaxCacheLen,
            generatorFactory: generatorFactory);

        if (agentBuilder is not null)
        {
            session.ToolDefinitionsJson = OpenAiMapper.ToolsJson(request.Tools);
            session.ProcessToolRequest = (_, _, _) => Task.FromResult(ToolRequestResult.ReturnToCaller());
        }

        // Initialize before replaying history so the agent/tool system prompt is
        // inserted first rather than appended after the conversation.
        session.InitializeChat();

        // Map OpenAI params -> session properties
        OpenAiMapper.ApplyToSession(request, session);
        session.MtpDraft = request.Mtp ?? (request.MtpTokens is not null || _options.MtpDraft);
        if (request.MtpTokens is null) session.MtpDraftTokens = _options.MtpDraftTokens;

        // Set context window.
        //
        // MaxTokens is the CONTEXT WINDOW; MaxNewTokens is the GENERATION limit.
        // This used to clamp the window to `Math.Min(MaxNewTokens, cacheLen)`,
        // which made ChatSession.TrimToFitContext compute
        // `contextBudget = MaxTokens - MaxNewTokens = 0`, fall back to
        // `MaxTokens / 2`, and cut the prompt down to a handful of tokens - so
        // every completion answered a prompt the caller never sent (and
        // `max_tokens: 1` trimmed to zero and threw "Prompt produced no token
        // IDs"). The two values are not interchangeable: leaving room for the
        // response is already TrimToFitContext's job.
        //
        // A batched session is bounded by its row, which already had the memory
        // budget divided across the batch.
        session.MaxTokens = batcher?.MaxSequenceLength
            ?? ModelConfig.ComputeMaxCacheLength(loaded.Model.Config, _options.MaxCacheLen);

        // Replay conversation history - all messages EXCEPT the last user
        // message. The last user message is returned separately so the caller
        // can pass it to GetResponseStreamAsync (which adds it to history
        // internally).
        var (systemPrompt, history) = OpenAiMapper.ToChatHistory(request.Messages);

        // A trailing tool result (or assistant tool-call turn) means the client
        // is resuming an existing tool loop: replay the whole history and
        // generate from it without synthesizing a new user message.
        // The last message is either a tool result, or an assistant turn that
        // carries tool calls (its results may still be pending).
        var last = request.Messages.LastOrDefault();
        bool continueFromHistory = last is not null
            && (last.Role is "tool" || last is AssistantMessage { ToolCalls.Count: > 0 });

        int lastUserIdx = -1;
        if (!continueFromHistory)
        {
            for (int i = history.Count - 1; i >= 0; i--)
            {
                if (history[i].Role == ChatRole.User)
                {
                    lastUserIdx = i;
                    break;
                }
            }
        }

        string lastUserMessage = lastUserIdx >= 0 ? history[lastUserIdx].Content ?? "" : "";

        var replay = new List<ChatMessage>(history.Count);
        for (int i = 0; i < history.Count; i++)
            if (i != lastUserIdx)
                replay.Add(history[i]);

        var desired = new List<ChatMessage>(replay.Count + 1);
        if (!string.IsNullOrWhiteSpace(systemPrompt))
            desired.Add(new ChatMessage { Role = ChatRole.System, Content = systemPrompt });
        desired.AddRange(replay);

        // Restore a cached prefix when this request extends one, then warm up: the
        // session imports the KV the snapshot covers and prefills only the rest.
        // A batched session owns no KV (the batcher owns the rows), so restoring
        // one would write into another sequence's state; skip it entirely.
        var cached = batcher is null ? _prefixCache.Take(loaded.ModelId, replay) : null;
        if (cached is not null)
        {
            session.RestorePrefix(cached.Snapshot, desired);
            await session.WarmupPrefillAsync();
        }
        else
        {
            foreach (var message in desired)
                session.AddMessage(message);
        }

        return (session, lastUserMessage, continueFromHistory);
    }
}
