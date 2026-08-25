using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;
using SharpMind.Model;
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
    public (IChatSession session, string lastUserMessage) CreateSession(LoadedModel loaded, CreateChatCompletionRequest request)
    {
        // Build the session via the standard factory
        var session = ChatSessionFactory.CreateChatSession(
            typeof(StandardGeneratorBuilder<KVCacherBuilder>),
            typeof(KVCacherBuilder),
            loaded.Model,
            loaded.Tokenizer,
            loaded.Meta,
            agentBuilder: null,
            preProcessor: null,
            postProcessor: null,
            progress: null,
            permissions: BuildPermissionCallback(),
            formatter: null,
            seed: null);

        // Map OpenAI params → session properties
        OpenAiMapper.ApplyToSession(request, session);

        // Set context window
        int effectiveCacheLen = loaded.Model.Config.EffectiveInferenceCacheLength;
        session.MaxTokens = session.MaxNewTokens > 0
            ? Math.Min(session.MaxNewTokens, effectiveCacheLen)
            : effectiveCacheLen;

        // Replay conversation history — all messages EXCEPT the last user
        // message. The last user message is returned separately so the caller
        // can pass it to GetResponseStreamAsync (which adds it to history
        // internally).
        var (systemPrompt, history) = OpenAiMapper.ToChatHistory(request.Messages);

        if (!string.IsNullOrWhiteSpace(systemPrompt))
            session.AddMessage(ChatRole.System, systemPrompt);

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
                continue; // skip — GetResponseStreamAsync will add it
            }
            session.AddMessage(history[i]);
        }

        return (session, lastUserMessage);
    }
}
