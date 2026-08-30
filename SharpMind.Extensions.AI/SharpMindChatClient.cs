using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Channels;
using Microsoft.Extensions.AI;
using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat;

namespace SharpMind.Extensions.AI;

/// <summary>
/// Adapts a SharpMind <see cref="IChatSession"/> into a standard
/// <see cref="IChatClient"/> for use with the Microsoft.Extensions.AI ecosystem.
/// <para>
/// The caller owns session creation and disposal. This adapter streams
/// responses back to MEAI consumers via <see cref="IAsyncEnumerable{T}"/>
/// and routes MEAI-provided tools through SharpMind's agent layer.
/// </para>
/// </summary>
public sealed class SharpMindChatClient : IChatClient, IAsyncDisposable
{
    private readonly IChatSession _session;
    private readonly IAgentBuilder? _agentBuilder;
    private readonly AiFunctionToolAdapter _toolAdapter = new();
    private bool _toolsRegistered;

    /// <summary>
    /// Creates a new adapter wrapping an existing SharpMind session.
    /// </summary>
    /// <param name="session">
    /// The SharpMind chat session to drive. Must already have
    /// <see cref="IChatSession.InitializeChat"/> called before the first
    /// <see cref="GetResponseAsync"/> or
    /// <see cref="GetStreamingResponseAsync"/> call.
    /// </param>
    /// <param name="agentBuilder">
    /// Optional agent builder for tool registration. When provided, MEAI
    /// tools from <see cref="ChatOptions.Tools"/> are registered on first use.
    /// </param>
    public SharpMindChatClient(IChatSession session, IAgentBuilder? agentBuilder = null)
    {
        ArgumentNullException.ThrowIfNull(session);
        _session = session;
        _agentBuilder = agentBuilder;

        // Single seam for tool dispatch: MEAI functions are invoked here
        // (the host owns the loop for them, so caching/telemetry/approval
        // wrappers can hang off an individual call); everything else defers
        // to the session's native agent loop, which keeps its File/Network
        // gating. Callers may replace this for a fully host-owned loop, e.g.
        // to return calls to a FunctionInvokingChatClient middleware.
        session.ProcessToolRequest = _toolAdapter.DispatchAsync;
    }

    /// <inheritdoc/>
    public Task<ChatResponse> GetResponseAsync(
        IEnumerable<MeChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return GetResponseCoreAsync(messages, options, cancellationToken);
    }

    private async Task<ChatResponse> GetResponseCoreAsync(
        IEnumerable<MeChatMessage> messages,
        ChatOptions? options,
        CancellationToken ct)
    {
        // Both paths must strip tool-call markup and surface returned calls
        // identically, so there is one implementation and the non-streaming
        // form aggregates it.
        return await GetStreamingResponseCoreAsync(messages, options, ct)
            .ToChatResponseAsync(ct);
    }

    /// <inheritdoc/>
    public IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(
        IEnumerable<MeChatMessage> messages,
        ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        return GetStreamingResponseCoreAsync(messages, options, cancellationToken);
    }

    private async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseCoreAsync(
        IEnumerable<MeChatMessage> messages,
        ChatOptions? options,
        [EnumeratorCancellation] CancellationToken ct)
    {
        ApplyOptions(options);
        RegisterToolsOnce(options);

        var messageList = messages.ToList();
        var lastUser = FindLastUserMessage(messageList);
        if (lastUser is null) yield break;

        _session.InitializeChat();

        var channel = Channel.CreateUnbounded<ChatStreamEntry>();

        var startTask = _session.StartChatAsync(
            () => new SmChatMessage { Content = lastUser, Role = SmChatRole.User, Name = "User" },
            entry =>
            {
                if (entry.Status == ChatStatus.Interrupted && entry.Error is not null)
                    throw new InvalidOperationException($"SharpMind turn failed: {entry.Error}");

                channel.Writer.TryWrite(entry);
            },
            ct);

        var readTask = Task.Run(async () =>
        {
            try { await startTask; }
            catch (OperationCanceledException) { }
            finally { channel.Writer.TryComplete(); }
        }, CancellationToken.None);

        string responseId = Guid.NewGuid().ToString("N");
        string messageId = Guid.NewGuid().ToString("N");
        ChatResponseUpdate Update(AIContent content) =>
            new(MeChatRole.Assistant, [content]) { ResponseId = responseId, MessageId = messageId };

        var pending = new StringBuilder();
        bool returnedCall = false;
        ChatFinishReason? finish = null;

        await foreach (var entry in channel.Reader.ReadAllAsync(ct))
        {
            // A call handed back by ToolRequestOutcome.ReturnToCaller. The markup
            // that produced it is still sitting in `pending`; drop it and surface
            // the call itself, which is what a tool middleware is waiting for.
            if (entry.Status == ChatStatus.ToolCall && entry.ToolCall is { } toolCall)
            {
                pending.Clear();
                returnedCall = true;
                finish = ChatFinishReason.ToolCalls;
                yield return Update(ChatMessageConverter.ToFunctionCall(toolCall));
                continue;
            }

            if (entry.Token is { Length: > 0 } tok)
            {
                pending.Append(tok);
                if (TakeEmittable(pending) is { Length: > 0 } emit)
                    yield return Update(new TextContent(emit));
            }

            if (entry.Status == ChatStatus.Complete)
                break;
        }

        await readTask;

        // Anything still held is either a partial "<tool_call>" prefix or a call
        // the model never closed; trailing prose is released, markup is not.
        if (!returnedCall && pending.Length > 0)
        {
            string tail = pending.ToString();
            if (!tail.StartsWith(ToolOpen, StringComparison.Ordinal))
                yield return Update(new TextContent(tail));
        }

        yield return new ChatResponseUpdate(MeChatRole.Assistant, (string?)null)
        {
            ResponseId = responseId,
            MessageId = messageId,
            FinishReason = finish ?? ChatFinishReason.Stop
        };
    }

    /// <inheritdoc/>
    public object? GetService(Type serviceType, object? key = null)
    {
        if (serviceType == typeof(IChatSession) || serviceType == typeof(SharpMindChatClient))
            return this;
        return null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _session.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }

    /// <inheritdoc/>
    public async ValueTask DisposeAsync()
    {
        await _session.DisposeAsync();
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    private void ApplyOptions(ChatOptions? options)
    {
        if (options is null) return;
        ChatMessageConverter.ApplyOptions(options, _session);
    }

    private void RegisterToolsOnce(ChatOptions? options)
    {
        if (_agentBuilder is null || _toolsRegistered) return;
        if (options?.Tools is not { Count: > 0 }) return;
        _toolAdapter.RegisterTools(_agentBuilder, options.Tools);
        _toolsRegistered = true;
    }

    private const string ToolOpen = "<tool_call>";
    private const string ToolClose = "</tool_call>";

    /// <summary>
    /// Takes the text that is safe to emit now out of <paramref name="pending"/>,
    /// leaving behind anything that is, or might still become, tool-call markup.
    /// <para>
    /// The session streams every generated fragment as it arrives and only parses
    /// the completed buffer for a tool call afterwards, so the markup reaches this
    /// adapter as ordinary text. A <b>completed</b> block means the session's own
    /// agent loop dispatched the tool (the default wiring) — the MEAI caller should
    /// see the reply that follows, never the call. An <b>unterminated</b> one is
    /// held: it may still be completed by the next fragment.
    /// </para>
    /// <para>
    /// ponytail: recognises the tagged form only, which is what
    /// <c>AgentBuilder</c>'s tool prompt asks the model for. A model that emits a
    /// bare <c>{"tool": ...}</c> object without the tag still has it dispatched
    /// correctly by the session; only the suppression here misses it.
    /// </para>
    /// </summary>
    private static string TakeEmittable(StringBuilder pending)
    {
        string s = pending.ToString();
        var emit = new StringBuilder();
        int i = 0;

        while (true)
        {
            int open = s.IndexOf(ToolOpen, i, StringComparison.Ordinal);
            if (open < 0) break;

            emit.Append(s, i, open - i);

            int close = s.IndexOf(ToolClose, open, StringComparison.Ordinal);
            if (close < 0)
            {
                // Unterminated — hold from the opening tag on.
                pending.Clear();
                pending.Append(s, open, s.Length - open);
                return emit.ToString();
            }

            i = close + ToolClose.Length;   // drop the whole block
        }

        // No open tag left. Hold back only a trailing partial "<tool_call>" prefix.
        int hold = TrailingPartialOpenLength(s);
        emit.Append(s, i, s.Length - i - hold);
        pending.Clear();
        pending.Append(s, s.Length - hold, hold);
        return emit.ToString();
    }

    /// <summary>
    /// Length of the suffix of <paramref name="s"/> that is a proper prefix of
    /// <see cref="ToolOpen"/> — the part that must be held because the next
    /// fragment could complete an opening tag.
    /// </summary>
    private static int TrailingPartialOpenLength(string s)
    {
        int max = Math.Min(ToolOpen.Length - 1, s.Length);
        for (int len = max; len > 0; len--)
            if (string.CompareOrdinal(s, s.Length - len, ToolOpen, 0, len) == 0)
                return len;
        return 0;
    }

    private static string? FindLastUserMessage(IList<MeChatMessage> messages)
    {
        for (int i = messages.Count - 1; i >= 0; i--)
        {
            if (messages[i].Role.Value == "user" || messages[i].Role == MeChatRole.User)
                return messages[i].Text;
        }
        return null;
    }
}
