using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Server.Protocol;

namespace SharpMind.Server;

/// <summary>
/// Creates IChatSession instances from a loaded model and an OpenAI chat
/// completion request. Each request gets its own session (the KV-cache is
/// rebuilt from the conversation history).
/// </summary>
public sealed class SessionFactory(SharpMindServerOptions options)
{
    private readonly SharpMindServerOptions _options = options;

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
    /// is returned separately — it must be passed to
    /// <see cref="IChatSession.GetResponseStreamAsync"/> to generate a response.
    /// </summary>
    public (IChatSession session, string lastUserMessage, bool continueFromHistory) CreateSession(LoadedModel loaded, CreateChatCompletionRequest request)
    {
        // OpenAI tools are executed by the API client, not the server. Register
        // them so the model sees their schema, then hand every call back.
        var agentBuilder = OpenAiMapper.CreateToolAgent(request.Tools);

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
            maxCacheLen: _options.MaxCacheLen);

        if (agentBuilder is not null)
        {
            session.ToolDefinitionsJson = OpenAiMapper.ToolsJson(request.Tools);
            session.ProcessToolRequest = (_, _, _) => Task.FromResult(ToolRequestResult.ReturnToCaller());
        }

        // Initialize before replaying history so the agent/tool system prompt is
        // inserted first rather than appended after the conversation.
        session.InitializeChat();

        // Map OpenAI params �+' session properties
        OpenAiMapper.ApplyToSession(request, session);

        // Set context window.
        //
        // MaxTokens is the CONTEXT WINDOW; MaxNewTokens is the GENERATION limit.
        // This used to clamp the window to `Math.Min(MaxNewTokens, cacheLen)`,
        // which made ChatSession.TrimToFitContext compute
        // `contextBudget = MaxTokens - MaxNewTokens = 0`, fall back to
        // `MaxTokens / 2`, and cut the prompt down to a handful of tokens — so
        // every completion answered a prompt the caller never sent (and
        // `max_tokens: 1` trimmed to zero and threw "Prompt produced no token
        // IDs"). The two values are not interchangeable: leaving room for the
        // response is already TrimToFitContext's job.
        session.MaxTokens = ModelConfig.ComputeMaxCacheLength(loaded.Model.Config, _options.MaxCacheLen);

        // Replay conversation history — all messages EXCEPT the last user
        // message. The last user message is returned separately so the caller
        // can pass it to GetResponseStreamAsync (which adds it to history
        // internally).
        var (systemPrompt, history) = OpenAiMapper.ToChatHistory(request.Messages);

        if (!string.IsNullOrWhiteSpace(systemPrompt))
            session.AddMessage(ChatRole.System, systemPrompt);

        // A trailing tool result (or assistant tool-call turn) means the client
        // is resuming an existing tool loop: replay the whole history and
        // generate from it without synthesizing a new user message.
        bool continueFromHistory = request.Messages.Count > 0
            && request.Messages[^1].Role is "tool"
            || request.Messages.LastOrDefault() is AssistantMessage { ToolCalls.Count: > 0 };

        if (continueFromHistory)
        {
            foreach (var message in history)
                session.AddMessage(message);
            return (session, "", true);
        }

        // Find the last user message index
        int lastUserIdx = -1;
        for (int i = history.Count - 1; i >= 0; i--)
        {
            if (history[i].Role == ChatRole.User)
            {
                lastUserIdx = i;
                break;
            }
        }

        string lastUserMessage = "";
        for (int i = 0; i < history.Count; i++)
        {
            if (i == lastUserIdx)
            {
                lastUserMessage = history[i].Content ?? "";
                continue; // skip - GetResponseStreamAsync will add it
            }
            session.AddMessage(history[i]);
        }

        return (session, lastUserMessage, false);
    }
}
