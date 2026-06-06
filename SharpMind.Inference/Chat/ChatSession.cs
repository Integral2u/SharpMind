using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace SharpMind.Inference.Chat;

public sealed class ChatSession<T, K> : IChatSession where K : IKVCacheBuilder, new() where T : IGeneratorBuilder<K>, new()
{
    private readonly Tokenizer _tokenizer;
    private readonly IGenerator<K> _generator;
    private readonly Transformer _model;
    private readonly List<ChatMessage> _history = [];
    private readonly IChatPromptFormatter? _formatter;
    private readonly IAgentBuilder? _agentBuilder;
    private readonly bool _addBos;
    private readonly bool _addEos;
    private bool _disposed;
    private int[]? _cachedPromptTokens;
    // ── Permission gate ──────────────────────────────────────────────────────
    /// <summary>
    /// Optional callback invoked before every tool call that originates from
    /// <see cref="IAgentBuilder.CallToolAsync"/>. The host application inspects
    /// <see cref="ToolPermissionContext"/> and returns:
    /// <list type="bullet">
    ///   <item><see cref="ToolPermission.Always"/> — execute immediately.</item>
    ///   <item><see cref="ToolPermission.Ask"/>   — block until the user confirms
    ///         (the callback itself is responsible for surfacing UI and resolving
    ///         to Always or Never before returning).</item>
    ///   <item><see cref="ToolPermission.Never"/> — deny the call; the model
    ///         receives an error result and may try another approach.</item>
    /// </list>
    /// When <see langword="null"/> all tools are executed without prompting
    /// (equivalent to returning <see cref="ToolPermission.Always"/> for every call).
    /// </summary>
    public Func<ToolPermissionContext, Task<ToolPermission>>? PermissionCallback { get; set; }

    /// <summary>
    /// Maximum number of consecutive tool calls allowed in a single agent turn
    /// before the loop is broken and the last tool result is returned as-is.
    /// Prevents runaway agentic loops. Default: 10.
    /// </summary>
    public int MaxToolCallsPerTurn { get; set; } = 10;
    public ChatSession(
        Transformer model,
        Tokenizer tokenizer,
        ModelMetaData? meta = null,
        IAgentBuilder? agentBuilder = null,
        IKVCache[]? caches = null,
        int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        _tokenizer = tokenizer;
        _formatter = ChatPromptFormatterFactory.Create(meta);
        _agentBuilder = agentBuilder;
        _addBos = meta?.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
        _addEos = meta?.GetLong("tokenizer.ggml.add_eos_token", 1) != 0;
        _generator = new T().CreateGenerator(model, tokenizer, _addBos, _addEos, caches, seed);
        ArgumentNullException.ThrowIfNull(_generator);

        if (_agentBuilder != null) AddMessage(ChatRole.System, _agentBuilder.BuildAgentPrompt());
    }

    public IGenerator<K> Generator => _generator;
    public Tokenizer Tokenizer => _tokenizer;
    public Transformer Model => _model;
    public IReadOnlyList<ChatMessage> History => _history.Where(p => p.Role != ChatRole.System).ToList();

    public int MaxTokens { get; set; } = 2048;
    public float Temperature { get; set; } = 0.0f;
    public int TopK { get; set; } = 20;
    public float TopP { get; set; } = 0.85f;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public int RepetitionWindow { get; set; } = 32;
    public float? TokensPerSecond { get; private set; }
    public float? TimeToFirstToken { get; private set; }


    public void AddMessage(ChatRole role, string content)
    {
        ThrowIfDisposed();
        _history.Add(new ChatMessage { Role = role, Content = content });
    }
    public void AddMessage(ChatMessage message)
    {
        ThrowIfDisposed();
        _history.Add(message);
    }

    public string GetFormattedPrompt()
    {
        ThrowIfDisposed();
        return BuildPrompt();
    }

    public void ClearHistory()
    {
        _history.Clear();
        if (_agentBuilder != null) AddMessage(ChatRole.System, _agentBuilder.BuildAgentPrompt());
        _cachedPromptTokens = null;
        _generator.ResetCache();
    }

    public void ResetCaches() => _generator.ResetCache();
    //review: should use prompt formatter?
    private string BuildPrompt()
    {
        if (_formatter is not null)
            return _formatter.Format(_history, _tokenizer, _addBos);

        var sb = new System.Text.StringBuilder();

        if (_addBos && _tokenizer.BosId >= 0)
            sb.Append(_tokenizer.IdToToken(_tokenizer.BosId));

        foreach (var msg in _history)
        {
            var prefix = msg.Role switch
            {
                ChatRole.System => "system: ",
                ChatRole.Agent => "assistant: ",
                ChatRole.User => "user: ",
                _ => ""
            };
            sb.Append(prefix);
            sb.Append(msg.Content);
            sb.Append('\n');
        }
        sb.Append("assistant: ");
        return sb.ToString();
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        await Task.CompletedTask;
        _generator.Dispose();
        _model.Dispose();
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(_disposed, typeof(ChatSession<T, K>).Name);
    // ── Tool call detection & dispatch ───────────────────────────────────────

    /// <summary>
    /// Returns true when <paramref name="text"/> looks like a well-formed agent
    /// tool-call JSON object, i.e. has both a <c>tool</c> string field and an
    /// <c>arguments</c> object field. Sets <paramref name="parsed"/> on success.
    /// </summary>
    private static bool TryParseToolCall(string text, out JsonObject? parsed)
    {
        parsed = null;
        var trimmed = text.Trim();
        if (trimmed.Length == 0 || trimmed[0] != '{') return false;

        try
        {
            var node = JsonNode.Parse(trimmed);
            if (node is not JsonObject obj) return false;
            if (obj["tool"]?.GetValueKind() != JsonValueKind.String) return false;
            if (obj["arguments"] is not JsonObject) return false;
            parsed = obj;
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>
    /// Determines the <see cref="ToolCategory"/> for a tool call by looking up
    /// the host object's marker interface via <see cref="IAgentBuilder.GetToolHost"/>.
    /// Falls back to <see cref="ToolCategory.General"/> when no IO marker is present
    /// or when the builder does not implement <c>GetToolHost</c>.
    /// </summary>
    private ToolCategory ResolveCategory(string toolName)
    {
        var host = _agentBuilder?.GetToolHost(toolName);
        if (host is INetworkToolService) return ToolCategory.Network;
        if (host is IFileToolService) return ToolCategory.File;
        return ToolCategory.General;
    }

    /// <summary>
    /// Checks the permission callback for <paramref name="toolName"/> with the
    /// supplied <paramref name="arguments"/>. Returns <see langword="true"/> when
    /// the call is allowed to proceed.
    /// </summary>
    private async Task<bool> IsPermittedAsync(
        string toolName, JsonObject arguments, CancellationToken ct)
    {
        if (PermissionCallback is null) return true;

        var ctx = new ToolPermissionContext
        {
            ToolName = toolName,
            Category = ResolveCategory(toolName),
            Arguments = arguments
        };

        // The callback handles its own "Ask" UI; by the time it returns it has
        // resolved to Always or Never.
        var permission = await PermissionCallback(ctx).WaitAsync(ct);
        return permission == ToolPermission.Always;
    }
    private async IAsyncEnumerable<ChatStreamEntry> GetResponseStreamAsync(
        string userInput,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ThrowIfDisposed();

        _history.Add(ChatMessage.User(userInput));

        // Agentic loop: keep generating until the model produces a plain response
        // rather than another tool call, or until MaxToolCallsPerTurn is reached.
        for (int toolCallCount = 0; ; toolCallCount++)
        {
            // ── Tokenise ─────────────────────────────────────────────────────
            int[] promptToks;
            if (_cachedPromptTokens is not null && _formatter is null)
            {
                // Incremental: previous prompt ended with "assistant: ".
                // Append <last-agent-response>\nuser: <input>\nassistant:
                var incremental = new System.Text.StringBuilder();
                if (_history.Count >= 2 && _history[^2].Role == ChatRole.Agent)
                    incremental.Append(_history[^2].Content).Append('\n');
                incremental.Append("user: ").Append(userInput).Append("\nassistant: ");

                int[] newToks = _tokenizer.Encode(incremental.ToString(), addBos: false, addEos: false);
                promptToks = GC.AllocateUninitializedArray<int>(_cachedPromptTokens.Length + newToks.Length);
                _cachedPromptTokens.CopyTo(promptToks.AsSpan());
                newToks.CopyTo(promptToks.AsSpan(_cachedPromptTokens.Length));
            }
            else
            {
                var prompt = BuildPrompt();
                promptToks = _tokenizer.Encode(prompt, addBos: false, addEos: false);
            }

            if (promptToks.Length > MaxTokens)
            {
                int start = promptToks.Length - MaxTokens;
                var subset = GC.AllocateUninitializedArray<int>(MaxTokens);
                promptToks.AsSpan(start, MaxTokens).CopyTo(subset);
                promptToks = subset;
            }

            if (promptToks.Length == 0)
                throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");

            _cachedPromptTokens = promptToks;
            _generator.ResetCache();

            var sampleCfg = new SamplingConfig
            {
                Temperature = Temperature,
                TopK = TopK,
                TopP = TopP,
            };

            var genCfg = new GenerationConfig
            {
                MaxNewTokens = MaxTokens,
                RepetitionPenalty = RepetitionPenalty,
                RepetitionWindow = RepetitionWindow,
                StopTokenIds = [_tokenizer.EosId],
                SlidingWindowSize = 0,
                Stream = true,
            };

            // ── Stream tokens ────────────────────────────────────────────────
            var response = new System.Text.StringBuilder();

            await foreach (var fragment in _generator.GenerateFromTokensAsync(promptToks, sampleCfg, genCfg, ct))
            {
                response.Append(fragment);

                // Safety: detect single-character repetition loop and stop
                if (response.Length >= 8)
                {
                    char last = response[^1];
                    bool loop = true;
                    for (int i = 2; i <= 8; i++)
                        if (response[^i] != last) { loop = false; break; }
                    if (loop) break;
                }

                yield return new ChatStreamEntry
                {
                    Status = ChatStatus.Responding,
                    Token = fragment,
                    IsComplete = false,
                    TokensPerSecond = _generator.TokensPerSecond,
                    TimeToFirstToken = _generator.TimeToFirstToken
                };
            }

            var responseText = response.ToString();

            // ── Tool call detection ──────────────────────────────────────────
            // Only attempt tool dispatch when an AgentBuilder is present and the
            // model actually produced a tool-call JSON.
            if (_agentBuilder is not null
                && toolCallCount < MaxToolCallsPerTurn
                && TryParseToolCall(responseText, out var toolCall)
                && toolCall is not null)
            {
                var toolName = toolCall["tool"]!.GetValue<string>();
                var args = toolCall["arguments"]!.AsObject();

                // Record the model's tool-call response in history as an agent turn
                // so the formatter can include it in the next prompt correctly.
                _history.Add(ChatMessage.Agent(responseText));

                // ── Permission gate ──────────────────────────────────────────
                bool permitted = await IsPermittedAsync(toolName, args, ct);

                JsonObject toolResult;
                if (!permitted)
                {
                    toolResult = new JsonObject
                    {
                        ["status"] = "error",
                        ["message"] = $"Permission denied for tool '{toolName}'."
                    };
                }
                else
                {
                    // Signal to the UI that a tool is executing
                    yield return new ChatStreamEntry
                    {
                        Status = ChatStatus.Executing,
                        Token = toolName,   // lets the UI display which tool is running
                        IsComplete = false,
                        TokensPerSecond = _generator.TokensPerSecond,
                        TimeToFirstToken = _generator.TimeToFirstToken
                    };

                    toolResult = await _agentBuilder.CallToolAsync(toolCall);
                }

                // Feed the tool result back as a system message so the model can
                // use it in its next generation pass.
                _history.Add(new ChatMessage
                {
                    Role = ChatRole.System,
                    Content = $"Tool result: {toolResult.ToJsonString()}"
                });

                // Invalidate the incremental cache — history has grown significantly
                _cachedPromptTokens = null;

                // Loop: generate again with the enriched history
                continue;
            }

            // ── Normal (non-tool) response ───────────────────────────────────
            if (responseText.Length > 0)
                _history.Add(ChatMessage.Agent(responseText));

            yield return new ChatStreamEntry
            {
                Status = ChatStatus.Complete,
                IsComplete = true,
                TokensPerSecond = _generator.TokensPerSecond,
                TimeToFirstToken = _generator.TimeToFirstToken
            };

            break; // Exit the agentic loop
        }
    }

    public async Task<ChatMessage[]> StartChatAsync(Func<Task<ChatMessage>> prompt, Action<ChatStreamEntry> response, CancellationToken token = default)
    {
        while (!token.IsCancellationRequested)
        {
            response(new ChatStreamEntry { Status = ChatStatus.Thinking, IsComplete = false });
            var input = await prompt();
            if (string.IsNullOrWhiteSpace(input.Content)) continue;
            try
            {
                await foreach (var entry in GetResponseStreamAsync(input.Content, token))
                {
                    response(entry);
                    TokensPerSecond = entry.TokensPerSecond;
                    TimeToFirstToken = entry.TimeToFirstToken;
                }
            }
            catch (OperationCanceledException)
            {
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted, IsComplete = true, TokensPerSecond = _generator.TokensPerSecond, TimeToFirstToken = _generator.TimeToFirstToken });
                break;
            }
            catch
            {
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted, IsComplete = true, TokensPerSecond = _generator.TokensPerSecond, TimeToFirstToken = _generator.TimeToFirstToken });
                break;
            }
        }
        return [.. _history];
    }

    public async Task<ChatMessage[]> StartChatAsync(Func<ChatMessage> prompt, Action<ChatStreamEntry> response, CancellationToken token = default)
        => await StartChatAsync(() => Task.FromResult(prompt()), response, token);

    public async Task<ChatMessage[]> StartChatAsync(Func<Task<string>> prompt, Action<string> response, CancellationToken token = default)
        => await StartChatAsync(async () => new ChatMessage { Content = await prompt(), Role = ChatRole.User }, (e) =>
        {
            if (e.Token is { Length: > 0 } delta) response(delta);
        }, token);

    public async Task<ChatMessage[]> StartChatAsync(Func<string> prompt, Action<string> response, CancellationToken token = default)
        => await StartChatAsync(() => new ChatMessage { Content = prompt(), Role = ChatRole.User }, (e) =>
        {
            if (e.Token is { Length: > 0 } delta) response(delta);
        }, token);
}
