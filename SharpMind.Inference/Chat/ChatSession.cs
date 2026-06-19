using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Threading.Channels;

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
    // IO interceptors (optional)
    // Supplied by the host application. When present and PermissionCallback is
    // set, they are activated only for the duration of each CallToolAsync call
    // so that ordinary session IO is never gated.
    private readonly InterceptingFileSystem? _fileSystem;
    private readonly InterceptingNetworkHandler? _networkHandler;
    private int _currentDepth;
    private string? _pendingDraft;
    private readonly System.Text.StringBuilder _responseBuffer = new();

    // Permission gate
    /// <summary>
    /// Callback invoked when a tool call attempts file system or network IO.
    /// When <see langword="null"/> <see cref="ToolPermission.Never"/> is returned.
    /// Receives a <see cref="ToolPermissionContext"/> describing the actual access
    /// attempt (path or URL, category, tool name, model arguments) and returns:
    /// <list type="bullet">
    ///   <item><see cref="ToolPermission.Always"/> — permit the access immediately.</item>
    ///   <item><see cref="ToolPermission.Ask"/>   — block until the user confirms.
    ///         The callback is responsible for surfacing UI and resolving to
    ///         <see cref="ToolPermission.Always"/> or <see cref="ToolPermission.Never"/>
    ///         before returning.</item>
    ///   <item><see cref="ToolPermission.Never"/> — deny the access; the tool receives
    ///         an <see cref="UnauthorizedAccessException"/> or
    ///         <see cref="System.Net.Http.HttpRequestException"/> and the model is given
    ///         an error result.</item>
    /// </list>
    /// </summary>
    public readonly Func<ToolPermissionContext, Task<ToolPermission>> PermissionCallback;

    /// <param name="fileSystem">
    /// Optional <see cref="InterceptingFileSystem"/> wrapping your real
    /// <c>System.IO.Abstractions.FileSystem</c>. When provided and
    /// <see cref="PermissionCallback"/> is set, every file-system access made by a
    /// tool call is gated through the callback.
    /// </param>
    /// <param name="networkHandler">
    /// Optional <see cref="InterceptingNetworkHandler"/> wrapping your real
    /// <see cref="HttpMessageHandler"/>. When provided and
    /// <see cref="PermissionCallback"/> is set, every outbound HTTP request made by a
    /// tool call is gated through the callback.
    /// </param>
    public int MaxToolCallsPerTurn { get; set; } = 10;
    /// <summary>Maximum sub-agent nesting depth. Default 2. Reached when both parent and one sub-agent are active.</summary>
    public int MaxAgentDepth { get; set; }
    public ChatSession(
        Transformer model,
        Tokenizer tokenizer,
        ModelMetaData? meta = null,
        IAgentBuilder? agentBuilder = null,
        Func<ToolPermissionContext, Task<ToolPermission>>? permissions = null,
        IKVCache[]? caches = null,
        int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        _tokenizer = tokenizer;
        _formatter = ChatPromptFormatterFactory.Create(meta);
        _agentBuilder = agentBuilder;
        MaxAgentDepth = _agentBuilder?.MaxAgentDepth ?? 2;
        _addBos = meta?.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
        _addEos = meta?.GetLong("tokenizer.ggml.add_eos_token", 1) != 0;
        _generator = new T().CreateGenerator(model, tokenizer, _addBos, _addEos, caches, seed);
        ArgumentNullException.ThrowIfNull(_generator);

        PermissionCallback = permissions ?? new Func<ToolPermissionContext, Task<ToolPermission>>(async (ctx) => { await Task.CompletedTask; return ToolPermission.Never; });
        _fileSystem = new InterceptingFileSystem();
        _networkHandler = new InterceptingNetworkHandler();

        if (_agentBuilder != null) AddMessage(ChatRole.System, _agentBuilder.BuildAgentPrompt());
    }

    public IGenerator<K> Generator => _generator;
    public Tokenizer Tokenizer => _tokenizer;
    public Transformer Model => _model;
    public IReadOnlyList<ChatMessage> History => [.. _history.Where(p => p.Role != ChatRole.System)];

    public int MaxTokens { get; set; } = 2048;
    public float Temperature { get; set; } = 0.0f;
    public int TopK { get; set; } = 20;
    public float TopP { get; set; } = 0.85f;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public int RepetitionWindow { get; set; } = 32;
    /// <summary>Token IDs that stop generation. Defaults to EOS if not set.</summary>
    public IReadOnlyList<int>? StopTokenIds { get; set; }
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
        _pendingDraft = null;
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

    private int[] TrimToFitContext(int[] promptToks)
    {
        if (promptToks.Length <= MaxTokens)
            return promptToks;

        // Phase 1: importance-scored message-level eviction
        var candidates = new List<(int Index, double Importance, DateTime Timestamp)>();
        for (int i = 0; i < _history.Count; i++)
        {
            var msg = _history[i];
            if (msg.IsPinned) continue;
            if (msg.Role == ChatRole.System && msg.Metadata?.TryGetValue("type", out var t) == true && t == "resume_draft")
                continue;

            double importance = 0.5;
            if (msg.Metadata?.TryGetValue("importance_score", out var s) == true)
                double.TryParse(s, out importance);

            candidates.Add((i, importance, msg.Timestamp));
        }

        candidates.Sort(static (a, b) =>
        {
            int cmp = a.Importance.CompareTo(b.Importance);
            return cmp != 0 ? cmp : a.Timestamp.CompareTo(b.Timestamp);
        });

        var removed = new HashSet<int>();
        foreach (var (idx, _, _) in candidates)
        {
            if (promptToks.Length <= MaxTokens) break;
            removed.Add(idx);

            var surviving = new List<ChatMessage>(_history.Count - removed.Count);
            for (int i = 0; i < _history.Count; i++)
                if (!removed.Contains(i))
                    surviving.Add(_history[i]);

            _history.Clear();
            _history.AddRange(surviving);
            _cachedPromptTokens = null;

            promptToks = _tokenizer.Encode(BuildPrompt(), addBos: false, addEos: false);
        }

        // Phase 2: token-level truncation from end as last resort
        if (promptToks.Length > MaxTokens)
        {
            int start = promptToks.Length - MaxTokens;
            var subset = GC.AllocateUninitializedArray<int>(MaxTokens);
            promptToks.AsSpan(start, MaxTokens).CopyTo(subset);
            promptToks = subset;
        }

        return promptToks;
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
    // Tool call detection & dispatch

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
        catch (JsonException) { return false; }
    }

    // Agent call tag parsing
    // Format: {{agent:<name>[:temp=<X>][:seed=<Y>]:<query>}}
    // Examples:
    //   {{agent:Athena-Alpha:research quantum computing}}
    //   {{agent:Hermes-Gamma:temp=0.7:seed=42:summarize this text}}

    private static bool TryParseAgentTag(string text, out string? name, out float? temperature, out int? seed, out string? query)
    {
        name = null; temperature = null; seed = null; query = null;
        var trimmed = text.Trim();
        if (!trimmed.StartsWith("{{agent:")) return false;
        if (!trimmed.EndsWith("}}")) return false;

        var inner = trimmed.AsSpan(8, trimmed.Length - 10); // strip "{{agent:" and "}}"

        int firstColon = inner.IndexOf(':');
        if (firstColon <= 0) return false;

        name = inner[..firstColon].ToString();

        // Parse optional params from middle segments; everything after params is the query
        int queryStart = firstColon + 1;
        while (queryStart < inner.Length)
        {
            int nextColon = inner[queryStart..].IndexOf(':');
            int segEnd = nextColon < 0 ? inner.Length : queryStart + nextColon;
            var segment = inner[queryStart..segEnd].ToString();

            if (segment.StartsWith("temp=") && float.TryParse(segment.AsSpan(5), out var t))
            {
                temperature = t;
                queryStart = segEnd + 1;
            }
            else if (segment.StartsWith("seed=") && int.TryParse(segment.AsSpan(5), out var s))
            {
                seed = s;
                queryStart = segEnd + 1;
            }
            else
            {
                query = inner[queryStart..].ToString();
                return true;
            }
        }

        // No query found
        return false;
    }

    private async Task<string> ExecuteSubAgentAsync(
        IAgent agent,
        string query,
        float? temperatureOverride,
        int? seedOverride,
        Action<string>? onSubFragment = null,
        CancellationToken ct = default)
    {
        // Build the sub-agent's prompt: system prompt + user query
        var prompt = $"{agent.Config.SystemPrompt}\n{query}";

        // Resolve temperature: override → agent config → tier default
        float temp = temperatureOverride ?? agent.Config.Temperature ?? 0.65f;
        int? seed = seedOverride ?? agent.Config.Seed;

        var sampleCfg = new SamplingConfig
        {
            Temperature = temp,
            TopK = TopK,
            TopP = TopP,
            Seed = seed,
        };

        var genCfg = new GenerationConfig
        {
            MaxNewTokens = MaxTokens,
            RepetitionPenalty = RepetitionPenalty,
            RepetitionWindow = RepetitionWindow,
            StopTokenIds = StopTokenIds ?? [_tokenizer.EosId],
            Stream = true,
        };

        var promptToks = _tokenizer.Encode(prompt, addBos: _addBos, addEos: false);

        _generator.ResetCache();

        var sb = new System.Text.StringBuilder();
        await foreach (var fragment in _generator.GenerateFromTokensAsync(promptToks, sampleCfg, genCfg, ct))
        {
            sb.Append(fragment);
            onSubFragment?.Invoke(fragment);
        }

        return sb.ToString();
    }

    // Tool dispatch with IO interception

    /// <summary>
    /// Activates the IO interceptors (if any) around <see cref="IAgentBuilder.CallToolAsync"/>,
    /// gating every file-system or network access through <see cref="PermissionCallback"/>.
    /// Interceptors are always deactivated in the finally block regardless of outcome.
    /// When <see cref="PermissionCallback"/> is null the interceptors are never activated.
    /// </summary>
    private async Task<JsonObject> DispatchToolAsync(
        string toolName, JsonObject toolCall, JsonObject args, CancellationToken ct)
    {
        async Task<bool> check(string tn, ToolCategory category, string resource, JsonObject callArgs)
        {
            var ctx = new ToolPermissionContext
            {
                ToolName = tn,
                Category = category,
                Resource = resource,
                Arguments = callArgs
            };
            var permission = await PermissionCallback(ctx).WaitAsync(ct);
            return permission == ToolPermission.Always;
        }

        // Activate interceptors only when we have a callback to wire them to
        if ((IoPermissionCheck?)check is not null)
        {
            _fileSystem?.Activate(toolName, args, check);
            _networkHandler?.Activate(toolName, args, check);
        }

        try
        {
            return await _agentBuilder!.CallToolAsync(toolCall);
        }
        finally
        {
            // Always deactivate — tool must not retain IO access after the call
            _fileSystem?.Deactivate();
            _networkHandler?.Deactivate();
        }
    }
    private async IAsyncEnumerable<ChatStreamEntry> GetResponseStreamAsync(
        string userInput,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ThrowIfDisposed();


        _history.Add(ChatMessage.User(userInput));

        // Agentic loop: keep generating until the model produces a plain response
        // rather than a tool call, or until MaxToolCallsPerTurn is reached.
        for (int toolCallCount = 0; ; toolCallCount++)
        {
            // Tokenise
            int[] promptToks;
            if (_cachedPromptTokens is not null && _formatter is null)
            {
                // Incremental encode: previous prompt ended with "assistant: "
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
                promptToks = TrimToFitContext(promptToks);

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
                StopTokenIds = StopTokenIds ?? [_tokenizer.EosId],
                SlidingWindowSize = 0,
                Stream = true,
            };

            // Stream tokens
            _responseBuffer.Clear();

            await foreach (var fragment in _generator.GenerateFromTokensAsync(promptToks, sampleCfg, genCfg, ct))
            {
                _responseBuffer.Append(fragment);

                // Safety: detect single-character repetition loop and stop
                if (_responseBuffer.Length >= 8)
                {
                    char last = _responseBuffer[^1];
                    bool loop = true;
                    for (int i = 2; i <= 8; i++)
                        if (_responseBuffer[^i] != last) { loop = false; break; }
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

            var responseText = _responseBuffer.ToString();

            // Tool call detection
            if (_agentBuilder is not null
                && toolCallCount < MaxToolCallsPerTurn
                && TryParseToolCall(responseText, out var toolCall)
                && toolCall is not null)
            {
                var toolName = toolCall["tool"]!.GetValue<string>();
                var args = toolCall["arguments"]!.AsObject();

                // Record the model's tool-call turn in history for the formatter
                _history.Add(ChatMessage.Agent(responseText));

                // Signal to the UI that a tool is about to execute
                yield return new ChatStreamEntry
                {
                    Status = ChatStatus.Executing,
                    Token = toolName,
                    IsComplete = false,
                    TokensPerSecond = _generator.TokensPerSecond,
                    TimeToFirstToken = _generator.TimeToFirstToken
                };

                // Dispatch with IO interception — interceptors gate any actual
                // file/network access the tool makes through PermissionCallback.
                // If PermissionCallback is null, interceptors are never activated.
                var toolResult = await DispatchToolAsync(toolName, toolCall, args, ct);

                // Feed the result back as a system message for the next generation pass
                _history.Add(new ChatMessage
                {
                    Role = ChatRole.System,
                    Content = $"Tool result: {toolResult.ToJsonString()}"
                });

                _cachedPromptTokens = null; // history grew; invalidate incremental cache
                continue;                   // generate again with enriched history
            }

            // Agent call detection ({{agent:...}} format)
            if (_agentBuilder is not null
                && _agentBuilder.AgentsEnabled
                && toolCallCount < MaxToolCallsPerTurn
                && _currentDepth < MaxAgentDepth
                && TryParseAgentTag(responseText, out var agentName, out var agentTemp, out var agentSeed, out var agentQuery)
                && agentName is not null && agentQuery is not null
                && _agentBuilder.RegisteredAgents.TryGetValue(agentName, out var subAgent))
            {
                // Record the model's agent-call turn in history
                _history.Add(ChatMessage.Agent(responseText));

                // Signal which agent is about to execute
                yield return new ChatStreamEntry
                {
                    Status = ChatStatus.Executing,
                    Token = agentName,
                    IsComplete = false,
                    TokensPerSecond = _generator.TokensPerSecond,
                    TimeToFirstToken = _generator.TimeToFirstToken
                };

                // Execute sub-agent with depth tracking, streaming Researching tokens
                _currentDepth++;
                var subChannel = Channel.CreateUnbounded<ChatStreamEntry>();

                async Task<string> RunSubAgentAsync()
                {
                    try
                    {
                        return await ExecuteSubAgentAsync(
                            subAgent, agentQuery, agentTemp, agentSeed,
                            fragment => subChannel.Writer.TryWrite(new ChatStreamEntry
                            {
                                Status = ChatStatus.Researching,
                                Token = fragment,
                                IsComplete = false,
                                TokensPerSecond = _generator.TokensPerSecond,
                                TimeToFirstToken = _generator.TimeToFirstToken
                            }),
                            ct);
                    }
                    finally
                    {
                        subChannel.Writer.TryComplete();
                    }
                }

                var subTask = RunSubAgentAsync();

                await foreach (var entry in subChannel.Reader.ReadAllAsync(ct))
                    yield return entry;

                string agentResult;
                try
                {
                    agentResult = await subTask;
                }
                finally
                {
                    _currentDepth--;
                }

                // Feed the result back as a system message
                _history.Add(new ChatMessage
                {
                    Role = ChatRole.System,
                    Content = $"Tool result: {agentResult}"
                });

                _cachedPromptTokens = null; // history grew; invalidate incremental cache
                continue;                   // generate again with enriched history
            }

            // Depth limit reached — inform the model
            if (_agentBuilder is not null
                && _agentBuilder.AgentsEnabled
                && toolCallCount < MaxToolCallsPerTurn
                && _currentDepth >= MaxAgentDepth
                && TryParseAgentTag(responseText, out _, out _, out _, out _))
            {
                _history.Add(ChatMessage.Agent(responseText));
                _history.Add(new ChatMessage
                {
                    Role = ChatRole.System,
                    Content = $"Tool result: {{\"status\":\"error\",\"message\":\"Maximum agent depth ({MaxAgentDepth}) reached. Cannot delegate further.\"}}"
                });
                _cachedPromptTokens = null;
                continue;
            }

            // Normal (non-tool) response
            if (responseText.Length > 0)
                _history.Add(ChatMessage.Agent(responseText));

            yield return new ChatStreamEntry
            {
                Status = ChatStatus.Complete,
                IsComplete = true,
                TokensPerSecond = _generator.TokensPerSecond,
                TimeToFirstToken = _generator.TimeToFirstToken
            };

            break;
        }
    }

    public async Task<ChatMessage[]> StartChatAsync(Func<Task<ChatMessage>> prompt, Action<ChatStreamEntry> response, CancellationToken token = default)
    {
        while (!token.IsCancellationRequested)
        {
            response(new ChatStreamEntry { Status = ChatStatus.Thinking, IsComplete = false });
            var input = await prompt();
            if (token.IsCancellationRequested)
            {
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted, IsComplete = true, TokensPerSecond = _generator.TokensPerSecond, TimeToFirstToken = _generator.TimeToFirstToken });
                break;
            }
            if (string.IsNullOrWhiteSpace(input.Content)) continue;

            // Soft recovery: inject pending draft from previous interruption
            if (_pendingDraft is not null)
            {
                var draft = _pendingDraft;
                _pendingDraft = null;
                AddMessage(new ChatMessage
                {
                    Role = ChatRole.System,
                    Content = $"[Resume from interruption]\nThe assistant was interrupted while generating:\n\n{draft}\n\nContinue seamlessly from where it left off.",
                    Metadata = new Dictionary<string, string> { ["type"] = "resume_draft" }
                });
            }

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
                if (_responseBuffer.Length > 0)
                    _pendingDraft = _responseBuffer.ToString();
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
