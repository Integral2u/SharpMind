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
        ApplyOptions(options);
        RegisterToolsOnce(options);

        var messageList = messages.ToList();
        var lastUser = FindLastUserMessage(messageList);
        if (lastUser is null)
            return new ChatResponse([]);

        _session.InitializeChat();

        var sb = new StringBuilder();
        var task = _session.StartChatAsync(
            () => new SmChatMessage { Content = lastUser, Role = SmChatRole.User, Name = "User" },
            entry =>
            {
                if (entry.Status == ChatStatus.Interrupted && entry.Error is not null)
                    throw new InvalidOperationException($"SharpMind turn failed: {entry.Error}");

                if (entry.Token is { Length: > 0 } tok)
                    sb.Append(tok);
            },
            ct);

        await task;

        var responseMsg = new MeChatMessage(MeChatRole.Assistant, sb.ToString());
        return new ChatResponse(responseMsg);
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

        await foreach (var entry in channel.Reader.ReadAllAsync(ct))
        {
            if (entry.Token is { Length: > 0 } tok)
                yield return new ChatResponseUpdate(MeChatRole.Assistant, tok);

            if (entry.Status == ChatStatus.Complete)
                break;
        }

        await readTask;
        yield return new ChatResponseUpdate(MeChatRole.Assistant, (string?)null)
        {
            FinishReason = ChatFinishReason.Stop
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
        AiFunctionToolAdapter.RegisterTools(_agentBuilder, options.Tools);
        _toolsRegistered = true;
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
