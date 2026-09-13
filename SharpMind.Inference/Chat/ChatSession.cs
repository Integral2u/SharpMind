using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Text.Json.Nodes;
using System.Threading.Channels;

namespace SharpMind.Inference.Chat;

public sealed class ChatSession<T, K> : IChatSession where K : IKVCacheBuilder, new() where T : IGeneratorBuilder<K>, new()
{
    private readonly Tokenizer _tokenizer;
    private IGenerator<K> _generator = null!;
    private readonly Transformer _model;
    private readonly bool _disposeModel;
    private readonly List<ChatMessage> _history = [];
    private IChatPromptFormatter? _formatter;
    private readonly IChatPromptFormatter? _formatterOverride;
    private readonly IAgentBuilder? _agentBuilder;
    private readonly bool _addBos;
    private readonly bool _addEos;
    private bool _initialized;
    private bool _disposed;
    /// <summary>
    /// Test/observation hook: how many tokens were actually fed to the generator
    /// as prefill on the most recent turn (the full prompt on a fallback, the
    /// delta slice when the KV cache was extended incrementally).
    /// </summary>
    public int LastPrefillTokenCount { get; private set; }
    private IReadOnlyList<ChatMessage>? _filteredHistoryCache;
    private IReadOnlyList<ChatMessage>? _promptHistoryCache;
    // IO interceptors (optional)
    // Supplied by the host application. When present and PermissionCallback is
    // set, they are activated only for the duration of each CallToolAsync call
    // so that ordinary session IO is never gated.
    private readonly InterceptingFileSystem? _fileSystem;
    private readonly InterceptingNetworkHandler? _networkHandler;
    private int _currentDepth;
    private string? _pendingDraft;
    private readonly System.Text.StringBuilder _responseBuffer = new();
    private bool _inThinkBlock;
    private readonly IPromptPreProcessor? _preProcessor;
    private readonly IPromptPostProcessor? _postProcessor;
    private readonly IProgress<float>? _progress;
    private readonly ModelMetaData? _meta;
    private readonly ToolCallFormat _toolCallFormat;

    /// <summary>
    /// Role used to feed a tool's result back into the history for the next
    /// generation pass. For native function-calling models (Qwen) the result is
    /// a <see cref="ChatRole.Tool"/> turn rendered as the template's tool block
    /// (<c>&lt;|im_start|&gt;tool</c>), which is what those models were trained to
    /// continue from; a mid-conversation system block reads as a stray system
    /// message and they respond as if no result ever arrived. Other formats keep
    /// the legacy system-message framing.
    /// </summary>
    private ChatRole ResultRole => _toolCallFormat is ToolCallFormat.Qwen or ToolCallFormat.Mistral
        ? ChatRole.Tool
        : ChatRole.System;

    /// <summary>
    /// Optional callback fired during the warm-up prefill in
    /// <see cref="InitializeChat"/> with the overall fraction (0..1) of the
    /// initial prompt prefilled. Lets the loading view surface "Prefilling X%"
    /// during session startup instead of jumping from 15% to done. Set null to
    /// suppress.
    /// </summary>
    internal Action<double>? PrefillProgress { get; set; }
    private readonly IKVCache[]? _caches;
    /// <summary>
    /// Optional accelerator-backed inference engine. When present,
    /// <see cref="InitializeChat"/> builds an <see cref="EngineGenerator{K}"/> around it
    /// instead of the generic builder, and session snapshots round-trip through the
    /// engine's whole-cache <see cref="IInferenceEngine.ExportCache"/>/
    /// <see cref="IInferenceEngine.ImportCache"/> APIs. Ownership follows the generator:
    /// <see cref="EngineGenerator{T}.Dispose"/> disposes the engine.
    /// </summary>
    private readonly IInferenceEngine? _engine;
    private readonly bool _hasEngine;

    /// <summary>
    /// Resolves the live KV cache array — either the explicitly-passed
    /// <see cref="_caches"/> or, when null (factory path), the caches
    /// owned by the generator after <see cref="InitializeChat"/>.
    /// </summary>
    private IKVCache[]? ActiveCaches => _caches ?? _generator?.Caches?.ToArray();

    private readonly int? _seed;
    private readonly int? _maxCacheLen;
    private string _userName = "User";
    private CancellationTokenSource? _turnCts;
    /// <summary>
    /// KV-cache snapshot restored from a saved session. Consumed (and
    /// nulled) by <see cref="WarmupPrefillAsync"/> when the prompt hash
    /// matches, allowing the prefill to be skipped entirely.
    /// </summary>
    private KVCacheSnapshot? _pendingKvCacheSnapshot;
    /// <summary>
    /// Fractions (0..1) reported by <see cref="IGenerator{T}.PrefillProgress"/>
    /// during the generator's synchronous chunked-prefill phase. Drained into
    /// <see cref="ChatStatus.Updating"/> stream entries at the top of the main
    /// generation loop so the UI can display "Prefilling NN.NN%".
    /// </summary>
    private readonly Queue<double> _prefillProgressQueue = [];


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

    /// <summary>
    /// Optional external tool-call interceptor. When set, every tool call the
    /// model makes is routed through this delegate before the native agent
    /// dispatches it — see <see cref="IChatSession.ProcessToolRequest"/> for
    /// the <see cref="ToolRequestOutcome"/> semantics. Null restores the
    /// built-in agent loop that dispatches every tool call itself.
    /// </summary>
    public Func<string, JsonObject, CancellationToken, Task<ToolRequestResult>>? ProcessToolRequest { get; set; }

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
    /// <param name="disposeModel">
    /// Whether disposing the session also disposes the <see cref="Transformer"/>
    /// passed in. Defaults to <see langword="false"/>: the caller constructed the
    /// model, so the caller owns it, and multiple sessions can share one loaded
    /// model. Pass <see langword="true"/> only to hand ownership over.
    /// </param>
    public int MaxToolCallsPerTurn { get; set; } = 10;
    /// <summary>Maximum sub-agent nesting depth. Default 2. Reached when both parent and one sub-agent are active.</summary>
    public int MaxAgentDepth { get; set; }
    public ChatSession(
        Transformer model,
        Tokenizer tokenizer,
        ModelMetaData? meta = null,
        IAgentBuilder? agentBuilder = null,
        IPromptPreProcessor? preProcessor = null,
        IPromptPostProcessor? postProcessor = null,
        IProgress<float>? progress = null,
        Func<ToolPermissionContext, Task<ToolPermission>>? permissions = null,
        IKVCache[]? caches = null,
        IChatPromptFormatter? formatter = null,
        int? seed = null,
        bool disposeModel = false,
        int? maxCacheLen = null,
        IInferenceEngine? engine = null,
        ToolCallFormat? toolCallFormat = null)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(model);
        _model = model;
        _tokenizer = tokenizer;
        _meta = meta;
        _agentBuilder = agentBuilder;
        _preProcessor = preProcessor;
        _postProcessor = postProcessor;
        _progress = progress;
        _caches = caches;
        _engine = engine;
        _hasEngine = engine is not null;
        _seed = seed;
        _disposeModel = disposeModel;
        _toolCallFormat = toolCallFormat ?? ToolCallFormatDetector.Resolve(_meta, _tokenizer);
        if (_agentBuilder is not null) _agentBuilder.CallFormat = _toolCallFormat;
        MaxAgentDepth = _agentBuilder?.MaxAgentDepth ?? 2;
        _addBos = ModelMetaData.ResolveAddBos(meta, tokenizer.UseSentencePieceMerge);
        _addEos = ModelMetaData.ResolveAddEos(meta);
        _formatterOverride = formatter;
        _maxCacheLen = maxCacheLen;
        PermissionCallback = permissions ?? new Func<ToolPermissionContext, Task<ToolPermission>>(async (ctx) => { await Task.CompletedTask; return ToolPermission.Never; });
        _fileSystem = new InterceptingFileSystem();
        _networkHandler = new InterceptingNetworkHandler();
    }

    public void InitializeChat(IProgress<float>? progress = null)
    {
        progress ??= _progress;
        if (_initialized) return;

        progress?.Report(0f);
        _formatter = _formatterOverride ?? ChatPromptFormatterFactory.Create(_meta, _tokenizer);
        progress?.Report(0.08f);

        progress?.Report(0.10f);
        _generator = _engine is null
            ? new T().CreateGenerator(_model, _tokenizer, _addBos, _addEos, _caches, _seed, _maxCacheLen)
            : new EngineGenerator<K>(_engine, _tokenizer, _addBos, _addEos, _model.Config.NumLayers, _seed);
        ArgumentNullException.ThrowIfNull(_generator);
        // Chunked-prefill progress: queued on the calling thread during the
        // generator's synchronous prefill phase, then drained into
        // ChatStatus.Updating entries at the start of the main stream loop so
        // the UI can show "Prefilling NN.NN%" instead of appearing stuck.
        _generator.PrefillProgress = p => _prefillProgressQueue.Enqueue(p);
        progress?.Report(0.40f);

        if (_agentBuilder != null) AddAgentMessages();
        progress?.Report(0.55f);

        // Warm up: encode the system prompt to ensure tokenizer is ready
        _tokenizer.Encode("", _addBos, _addEos);
        progress?.Report(0.60f);

        _initialized = true;
    }

    /// <summary>
    /// Prefills the initial prompt (system prompt + tools) into the generator's
    /// KV cache after <see cref="InitializeChat"/> completes. Reports chunked
    /// progress (0..1) via <see cref="PrefillProgress"/> so the loading view
    /// can surface "Prefilling X%" during startup.
    /// When a <see cref="KVCacheSnapshot"/> was restored via
    /// <see cref="LoadSnapshot"/> and its prompt hash matches the freshly-built
    /// prompt, the prefill is skipped and the cache is restored directly — the
    /// first user turn then extends it incrementally as usual.
    /// </summary>
    internal async Task WarmupPrefillAsync()
    {
        EnsureInitialized();
        string prompt = BuildPrompt();
        int[] promptToks = _tokenizer.Encode(prompt, _addBos, _addEos);
        if (promptToks.Length == 0) return;

        // Trim to the model's context window so small-context checkpoints
        // (e.g. MaxSeqLen=256 from training defaults) don't overflow RoPE.
        int maxContext = MaxTokens;
        if (maxContext > 0 && promptToks.Length > maxContext)
        {
            int start = promptToks.Length - maxContext;
            var trimmed = GC.AllocateUninitializedArray<int>(maxContext);
            promptToks.AsSpan(start, maxContext).CopyTo(trimmed);
            promptToks = trimmed;
        }

        // Fast path: if a cached KV-cache snapshot was loaded and the prompt
        // tokens hash matches, restore the cache directly — no prefill needed.
        if (_pendingKvCacheSnapshot is { } cached)
        {
            string currentHash = KVCacheSnapshot.HashPromptTokens(promptToks);
            bool shapeMatches = _hasEngine
                ? cached.Layers.Count == _model.Config.NumLayers
                : cached.Layers.Count == (ActiveCaches?.Length ?? 0);
            if (string.Equals(currentHash, cached.PromptHash, StringComparison.OrdinalIgnoreCase)
                && shapeMatches)
            {
                PrefillProgress?.Invoke(1.0);
                _generator!.ResetCache();
                if (_hasEngine && _engine is not null)
                    _engine.ImportCache(cached);
                else
                {
                    var ac = ActiveCaches!;
                    for (int i = 0; i < ac.Length && i < cached.Layers.Count; i++)
                        ac[i].RestoreBytes(cached.Layers[i]);
                }
                _generator!.SetCacheTokens(promptToks);
                _pendingKvCacheSnapshot = null;
                return;
            }
            // Hash mismatch — fall through to full prefill.
            _pendingKvCacheSnapshot = null;
        }

        _generator!.ResetCache();
        var warmupCfg = new SamplingConfig { Temperature = 0, TopK = 1 };
        var warmupGen = new GenerationConfig { MaxNewTokens = 0, Stream = false };
        _prefillProgressQueue.Clear();
        _generator.PrefillProgress = p =>
        {
            _prefillProgressQueue.Enqueue(p);
            PrefillProgress?.Invoke(p);
        };
        await foreach (var _ in _generator.GenerateFromTokensAsync(promptToks, warmupCfg, warmupGen)) { }
        _generator.PrefillProgress = p => _prefillProgressQueue.Enqueue(p);
    }

    private void EnsureInitialized()
    {
        if (!_initialized) InitializeChat();
    }

    public IGenerator<K> Generator
    {
        get
        {
            EnsureInitialized();
            return _generator!;
        }
    }
    public Tokenizer Tokenizer => _tokenizer;
    public Transformer Model => _model;
    public IReadOnlyList<ChatMessage> History => _filteredHistoryCache ??= [.. _history.Where(p => p.Role != ChatRole.System)];
    public IChatPromptFormatter? Formatter => _formatter;

    public int MaxTokens { get; set; } = 2048;
    /// <summary>Max generation tokens. Defaults to 256 so context trimming leaves room.</summary>
    public int MaxNewTokens { get; set; } = 256;
    /// <summary>
    /// Low values (0.1–0.3): Makes the AI predictable, focused, and accurate, which is best for facts or code.
    /// High values (0.8–1.5): Increases randomness and creativity, which is ideal for storytelling or brainstorming.
    /// </summary>
    public float Temperature { get; set; } = 0.0f;
    /// <summary>
    /// Top-K restricts the model's word choice to a fixed number (K) of the most likely next words.
    /// How it works: 
    /// If K = 40, the model picks only from the 40 highest-scoring words, throwing away all other words in its dictionary.
    /// Effect: A low K makes text predictable and safe. A high K allows for more variety, but can occasionally let in weird or off-topic words.
    /// </summary>
    public int TopK { get; set; } = 20;
    /// <summary>
    /// Top-P restricts word choice based on a cumulative probability threshold (P).
    /// How it works: If P = 0.90, the model adds up the probabilities of the best words from highest to lowest until the total reaches 90%, and only picks from that group.
    /// Effect: It is dynamic. When the AI is very sure of what to say next, the pool shrinks to just 1 or 2 words. 
    /// When the AI is unsure, the pool automatically grows to include more creative options.
    /// </summary>
    public float TopP { set; get; } = 0.85f;
    /// <summary>
    /// Reduces probability: Lowers the score of any word or token that already appeared.
    /// Values: Usually ranges from 1.0 (no penalty) up to 1.5 or higher. Higher numbers force more word variety.
    /// Downside: Setting it too high can break normal grammar and ruin punctuation
    /// </summary>
    public float RepetitionPenalty { get; set; } = 1.1f;
    /// <summary>
    /// Lookback limit: Counts the exact number of recent tokens the AI checks.
    /// Scope control: A window of 512 means the model checks the last 512 tokens generated for repeats.
    /// Setting it to 0 turns the window off, while setting it too high can slow down generation or penalize words needed for structural coherence.
    /// </summary>
    public int RepetitionWindow { get; set; } = 32;
    /// <summary>Token IDs that stop generation. Defaults to EOS if not set.</summary>
    public IReadOnlyList<int>? StopTokenIds { get; set; }
    /// <summary>Text strings that stop generation when matched in the decoded output.</summary>
    public IReadOnlyList<string>? StopStrings { get; set; }
    public bool ShowThinking { get; set; } = true;
    /// <summary>
    /// Value of the <c>enable_thinking</c> chat-template variable (checked by
    /// Qwen3-style templates). Defaults to <see langword="false"/> so those
    /// models emit an empty reasoning block and answer directly instead of
    /// streaming visible chain-of-thought before the response.
    /// </summary>
    public bool EnableThinking { get; set; }
    /// <summary>JSON array of tool/function definitions (from AgentBuilder.ToolDefinitions).</summary>
    public string? ToolDefinitionsJson { get; set; }
    public string UserName { get => _userName; set => _userName = value ?? "User"; }
    public float? TokensPerSecond { get; private set; }
    public float? TimeToFirstToken { get; private set; }


    public void AddMessage(ChatRole role, string content)
    {
        ThrowIfDisposed();
        _history.Add(new ChatMessage { Role = role, Content = content });
        InvalidateHistoryCache();
    }
    public void AddMessage(ChatMessage message)
    {
        ThrowIfDisposed();
        _history.Add(message);
        InvalidateHistoryCache();
    }

    /// <summary>
    /// Inserts the agent's standalone system prompts (embedded defaults) at the
    /// top of the history, then the synthesized agent prompt.
    /// </summary>
    private void AddAgentMessages()
    {
        ThrowIfDisposed();
        if (_agentBuilder == null) return;
        foreach (string prompt in _agentBuilder.AdditionalSystemPrompts)
        {
            _history.Add(new ChatMessage { Role = ChatRole.System, Content = prompt });
        }
        _history.Add(new ChatMessage { Role = ChatRole.System, Content = _agentBuilder.BuildAgentPrompt() });
        InvalidateHistoryCache();
    }

    public string GetFormattedPrompt()
    {
        ThrowIfDisposed();
        return BuildPrompt();
    }

    public void ClearHistory()
    {
        _history.Clear();
        InvalidateHistoryCache();
        _pendingDraft = null;
        if (_agentBuilder != null) AddAgentMessages();
        _generator.ResetCache();
    }

    public void ResetCaches()
    {
        EnsureInitialized();
        _generator.ResetCache();
    }

    /// <summary>
    /// Test hook: primes the generator's physical KV cache with
    /// <paramref name="prefixToks"/> so tests can drive the incremental-prefill
    /// path with a cache prefix of their choosing instead of relying on a
    /// tokenizer that round-trips sampled ids.
    /// </summary>
    internal async Task PrimeIncrementalCacheForTestAsync(int[] prefixToks)
    {
        EnsureInitialized();
        _generator.ResetCache();
        var sampleCfg = new SamplingConfig { Temperature = 0, TopK = 1 };
        var genCfg = new GenerationConfig { MaxNewTokens = 0, Stream = false };
        await foreach (var _ in _generator.GenerateFromTokensAsync(prefixToks, sampleCfg, genCfg)) { }
    }
    /// <summary>
    /// Returns true when the rendered prompt ends inside an unclosed &lt;think&gt;
    /// tag.  Used to seed <see cref="_inThinkBlock"/> before generation starts,
    /// since Qwen3 and DeepSeek-R1 templates emit &lt;think&gt;\n as part of the
    /// assistant prefix — the model never generates the opening tag, so a
    /// stream-only check would never fire.
    /// </summary>
    private bool PromptContainsUnclosedThinkTag()
    {
        string prompt = BuildPrompt();
        int lastOpen = prompt.LastIndexOf("<think>", StringComparison.Ordinal);
        int lastClose = prompt.LastIndexOf("</think>", StringComparison.Ordinal);
        return lastOpen > lastClose;
    }

    private string BuildPrompt()
    {
        if (_formatter is not null)
            return _formatter.Format(GetPromptHistory(), _tokenizer, _addBos, EnableThinking, ToolDefinitionsJson);

        var sb = new System.Text.StringBuilder();

        if (_addBos && _tokenizer.BosId >= 0)
            sb.Append(_tokenizer.IdToToken(_tokenizer.BosId));

        foreach (var msg in GetPromptHistory())
        {
            if (msg.Ignore) continue;
            var prefix = msg.Role switch
            {
                ChatRole.System => "system: ",
                ChatRole.Agent => "assistant: ",
                ChatRole.User => "user: ",
                ChatRole.Tool => "tool: ",
                _ => ""
            };
            sb.Append(prefix);
            sb.Append(msg.Content);
            sb.Append('\n');
        }
        sb.Append("assistant: ");
        return sb.ToString();
    }

    private IReadOnlyList<ChatMessage> GetPromptHistory()
    {
        if (EnableThinking)
            return _history;

        if (_promptHistoryCache is not null)
            return _promptHistoryCache;

        var list = new List<ChatMessage>(_history.Count);
        foreach (var msg in _history)
        {
            if (msg.Ignore) continue;
            if (msg.Role == ChatRole.Agent)
            {
                list.Add(new ChatMessage
                {
                    Role = msg.Role,
                    Content = ChatSession<T, K>.StripThinking(msg.Content),
                    Name = msg.Name,
                    Timestamp = msg.Timestamp,
                    IsPinned = msg.IsPinned,
                    Ignore = msg.Ignore,
                    Metadata = msg.Metadata,
                    Artifacts = msg.Artifacts
                });
            }
            else
            {
                list.Add(msg);
            }
        }
        _promptHistoryCache = list;
        return list;
    }

    private int[] TrimToFitContext(int[] promptToks)
    {
        int contextBudget = MaxTokens - MaxNewTokens;
        if (contextBudget <= 0) contextBudget = MaxTokens / 2;
        if (promptToks.Length <= contextBudget)
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
                if (double.TryParse(s, out var metaImportance)) importance = metaImportance;

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
            if (promptToks.Length <= contextBudget) break;
            if (!removed.Add(idx)) continue;

            var surviving = new List<ChatMessage>();
            for (int i = 0; i < _history.Count; i++)
                if (!removed.Contains(i))
                    surviving.Add(_history[i]);

            _history.Clear();
            _history.AddRange(surviving);
            InvalidateHistoryCache();

            promptToks = _tokenizer.Encode(BuildPrompt(), addBos: false, addEos: false);
        }

        // Phase 2: token-level truncation from end as last resort
        if (promptToks.Length > contextBudget)
        {
            int start = promptToks.Length - contextBudget;
            var subset = GC.AllocateUninitializedArray<int>(contextBudget);
            promptToks.AsSpan(start, contextBudget).CopyTo(subset);
            promptToks = subset;
            // Cutting tokens from the front means nothing resident in the KV
            // cache lines up with this any more; the prefix match in
            // FeedForPrompt sees that and rebuilds from scratch.
        }

        return promptToks;
    }

    /// <summary>
    /// Decides what to feed the generator for <paramref name="promptToks"/>, the
    /// full prompt as rendered now. Whatever the KV cache already holds for the
    /// leading tokens is kept and only the rest is fed: the cache is compared
    /// token by token against the prompt (<see cref="IGenerator{T}.CacheTokens"/>),
    /// truncated to the common prefix, and generation resumes from there.
    ///
    /// Comparing tokens rather than remembering "what we sent last turn" is what
    /// makes this hold for every formatter and every history edit at once. A
    /// second turn under ChatML extends the previous prompt plus the response the
    /// model itself generated, so almost everything matches. Compaction, an
    /// evicted message, thinking stripped from an assistant turn, or a response
    /// that re-tokenises differently from how it was generated all just move the
    /// divergence point earlier, and only what follows it is recomputed. A cache
    /// the generator cannot vouch for (null) or one with nothing in common is
    /// reset and the whole prompt is fed.
    ///
    /// At least one token is always fed, even when the cache already covers the
    /// whole prompt: the generator needs a forward pass to have logits to sample.
    /// </summary>
    private int[] FeedForPrompt(int[] promptToks)
    {
        var cached = _generator!.CacheTokens;
        int keep = 0;
        if (cached is not null)
        {
            int limit = Math.Min(cached.Count, promptToks.Length - 1);
            while (keep < limit && cached[keep] == promptToks[keep]) keep++;
        }

        if (keep == 0)
        {
            _generator.ResetCache();
            return promptToks;
        }

        _generator.TruncateCache(keep);
        return promptToks[keep..];
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        await Task.CompletedTask;
        _generator.Dispose();
        if (_disposeModel)
            _model.Dispose();
    }

private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(_disposed, typeof(ChatSession<T, K>).Name);
    private void InvalidateHistoryCache()
    {
        _filteredHistoryCache = null;
        _promptHistoryCache = null;
    }

    private static string StripThinking(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return text;

        // Remove all <think>...</think> blocks
        var result = RegexGenerated.ThinkingBlocks.Replace(text,string.Empty);// Regex.Replace(text, @"<think>.*?</think>", "", RegexOptions.Singleline);

        // Also trim trailing garbage characters often seen at the end of LLM responses (e.g. EOS tokens decoded as symbols)
        return result.TrimEnd('\uFFFD', '\u0000', '\u0001', '\u0002', '\u0003').Trim();
    }

    private static IReadOnlyList<string> MergeStopStrings(
        IReadOnlyList<string>? formatter, IReadOnlyList<string>? session)
    {
        if (session is { Count: > 0 } && formatter is { Count: > 0 })
            return formatter.Concat(session).Distinct().ToList();
        return session ?? formatter ?? [];
    }

    /// <summary>
    /// Returns true when <paramref name="text"/> contains a valid agent tool-call
    /// JSON object, i.e. has both a <c>tool</c> string field and an
    /// <c>arguments</c> object field. Accepts two formats:
    /// <list type="bullet">
    ///   <item><c>&lt;tool_call&gt;{"tool":"name","arguments":{...}}&lt;/tool_call&gt;</c></item>
    ///   <item><c>{"tool":"name","arguments":{...}}</c> — raw JSON</item>
    /// </list>
    /// Performs light repair on common model-output malformations (trailing
    /// commas, truncated brackets) before attempting to parse. Sets
    /// <paramref name="parsed"/> on success.
    /// </summary>
    private static bool TryParseToolCall(string text, ToolCallFormat format, out JsonObject? parsed)
    {
        parsed = null;
        if (format == ToolCallFormat.None) return false;
        var trimmed = text.Trim();
        if (trimmed.Length == 0) return false;

        // Qwen-instruct's native contract: a raw {"name":...,"arguments":...}
        // object with no wrapper. Look there first; tolerates the tagged wrapper
        // for a model that half-remembers the SharpMind shape.
        if (format == ToolCallFormat.Qwen)
        {
            if (trimmed[0] == '{')
            {
                int valueEnd = FindJsonValueEnd(trimmed, 0);
                if (valueEnd >= 0 && TryParseJsonObject(trimmed[..valueEnd], out parsed)) return true;
                // Cut off mid-generation (no closing brace) — the repair logic
                // in TryParseJsonObject finishes it.
                if (TryParseJsonObject(trimmed, out parsed)) return true;
            }
            if (TryParseTaggedBlock(trimmed, out parsed)) return true;
            return TryParseSloppyBlock(trimmed, out parsed);
        }

        // Mistral's native contract: [TOOL_CALLS]{"name":...,"arguments":...}
        if (format == ToolCallFormat.Mistral)
        {
            int marker = trimmed.IndexOf("[TOOL_CALLS]", StringComparison.Ordinal);
            if (marker >= 0)
            {
                int jsonStart = marker + "[TOOL_CALLS]".Length;
                while (jsonStart < trimmed.Length && char.IsWhiteSpace(trimmed[jsonStart]))
                    jsonStart++;
                if (jsonStart < trimmed.Length && trimmed[jsonStart] == '{')
                {
                    int valueEnd = FindJsonValueEnd(trimmed, jsonStart);
                    if (valueEnd >= 0 && TryParseJsonObject(trimmed[jsonStart..valueEnd], out parsed)) return true;
                    if (TryParseJsonObject(trimmed[jsonStart..], out parsed)) return true;
                }
            }
            // Tolerate a raw JSON object with no marker.
            if (trimmed[0] == '{')
            {
                int valueEnd = FindJsonValueEnd(trimmed, 0);
                if (valueEnd >= 0 && TryParseJsonObject(trimmed[..valueEnd], out parsed)) return true;
                if (TryParseJsonObject(trimmed, out parsed)) return true;
            }
            return false;
        }

        // Llama-3's native contract: a raw JSON object (often preceded by
        // <|python_tag|> but that token is stripped from the visible stream).
        // Tolerates the pipe-less <python_tag> variant a model may drift to.
        if (format == ToolCallFormat.Llama3)
        {
            int marker = trimmed.IndexOf("<|python_tag|>", StringComparison.Ordinal);
            if (marker < 0) marker = trimmed.IndexOf("<python_tag>", StringComparison.Ordinal);
            int markerLen = marker >= 0
                ? (trimmed.AsSpan(marker).StartsWith("<|python_tag|>") ? "<|python_tag|>".Length : "<python_tag>".Length)
                : 0;
            int start = marker >= 0 ? marker + markerLen : 0;
            while (start < trimmed.Length && char.IsWhiteSpace(trimmed[start]))
                start++;
            if (start < trimmed.Length && trimmed[start] == '{')
            {
                int valueEnd = FindJsonValueEnd(trimmed, start);
                if (valueEnd >= 0 && TryParseJsonObject(trimmed[start..valueEnd], out parsed)) return true;
                if (TryParseJsonObject(trimmed[start..], out parsed)) return true;
            }
            if (trimmed[0] == '{' && start != 0)
            {
                int valueEnd = FindJsonValueEnd(trimmed, 0);
                if (valueEnd >= 0 && TryParseJsonObject(trimmed[..valueEnd], out parsed)) return true;
                if (TryParseJsonObject(trimmed, out parsed)) return true;
            }
            // Tolerate a <tool_call> wrapper if the model drifts.
            if (TryParseTaggedBlock(trimmed, out parsed)) return true;
            return false;
        }

        // Gemma/functiongemma's native contract: a {"name":...,"args":...}
        // JSON object, optionally wrapped in <function_call> tags, or preceded
        // by a "functionCall" marker.
        if (format == ToolCallFormat.Gemma)
        {
            // <function_call>{...}</function_call> wrapper
            int fcStart = trimmed.IndexOf("<function_call>", StringComparison.Ordinal);
            if (fcStart >= 0)
            {
                int jsonStart = fcStart + "<function_call>".Length;
                while (jsonStart < trimmed.Length && char.IsWhiteSpace(trimmed[jsonStart]))
                    jsonStart++;
                int fcEnd = trimmed.IndexOf("</function_call>", StringComparison.Ordinal);
                int end = fcEnd >= 0 ? fcEnd : trimmed.Length;
                if (jsonStart < end && jsonStart < trimmed.Length && trimmed[jsonStart] == '{')
                    if (TryParseJsonObject(trimmed[jsonStart..end], out parsed)) return true;
            }
            // "functionCall" prefix marker (Gemini-style output)
            int fname = trimmed.IndexOf("functionCall", StringComparison.Ordinal);
            if (fname >= 0)
            {
                int jsonStart = trimmed.IndexOf('{', fname);
                if (jsonStart >= 0)
                {
                    int valueEnd = FindJsonValueEnd(trimmed, jsonStart);
                    if (valueEnd >= 0 && TryParseJsonObject(trimmed[jsonStart..valueEnd], out parsed)) return true;
                    if (TryParseJsonObject(trimmed[jsonStart..], out parsed)) return true;
                }
            }
            // Raw JSON object with "args" key
            if (trimmed[0] == '{')
            {
                int valueEnd = FindJsonValueEnd(trimmed, 0);
                if (valueEnd >= 0 && TryParseJsonObject(trimmed[..valueEnd], out parsed)) return true;
                if (TryParseJsonObject(trimmed, out parsed)) return true;
            }
            return false;
        }

        // SharpMind: the tagged contract first, then the sloppy
        // "<tool_call {json}" a small model drifts to, then a raw JSON object.
        if (TryParseTaggedBlock(trimmed, out parsed)) return true;
        if (TryParseSloppyBlock(trimmed, out parsed)) return true;
        if (trimmed[0] == '{') return TryParseJsonObject(trimmed, out parsed);
        return false;
    }

    private static bool TryParseTaggedBlock(string trimmed, out JsonObject? parsed)
    {
        parsed = null;
        var toolCallM = RegexGenerated.ToolCallBlocks.Match(trimmed);// Regex.Match(trimmed, @"<tool_call>(.*?)</tool_call>", RegexOptions.Singleline);
        if (!toolCallM.Success) return false;
        string inner = toolCallM.Groups[1].Value.Trim();
        return TryParseJsonObject(inner, out parsed);
    }

    private static bool TryParseSloppyBlock(string trimmed, out JsonObject? parsed)
    {
        parsed = null;
        // A model that starts a call as "<tool_call {json}" and never emits the
        // '>' or the closing tag. The first parseable tool-call JSON after the
        // <tool_call marker is the call; a narration prefix is fine.
        int marker = trimmed.IndexOf("<tool_call", StringComparison.Ordinal);
        if (marker < 0) return false;
        int jsonStart = marker + "<tool_call".Length;
        while (jsonStart < trimmed.Length && (char.IsWhiteSpace(trimmed[jsonStart]) || trimmed[jsonStart] == '>'))
            jsonStart++;
        if (jsonStart >= trimmed.Length || trimmed[jsonStart] != '{') return false;
        int valueEnd = FindJsonValueEnd(trimmed, jsonStart);
        if (valueEnd >= 0 && TryParseJsonObject(trimmed[jsonStart..valueEnd], out parsed)) return true;
        // The JSON was cut off mid-generation (no closing brace) — let
        // TryParseJsonObject's truncation repair finish it.
        return TryParseJsonObject(trimmed[jsonStart..], out parsed);
    }

    /// <summary>
    /// Strips model output that is tool machinery rather than prose:
    /// complete <c>&lt;tool_call&gt;</c> blocks (parsed above, or too malformed
    /// to parse) and a trailing unclosed <c>&lt;tool_call&gt;</c> span. Also
    /// strips the leading markers of native-function-calling models
    /// (<c>[TOOL_CALLS]</c>, <c>&lt;|python_tag|&gt;</c>,
    /// <c>&lt;function_call&gt;</c>) so their machinery cannot leak into the
    /// transcript as ordinary text.
    /// </summary>
    private static string SanitizeToolMarkup(string text)
    {
        string result = text;

        // Remove complete <tool_call>...</tool_call> blocks
        int tcIdx = result.IndexOf("<tool_call", StringComparison.Ordinal);
        if (tcIdx >= 0)
        {
            string stripped = RegexGenerated.ToolCallBlocks.Replace(result, string.Empty);
            int open = stripped.IndexOf("<tool_call", StringComparison.Ordinal);
            result = open < 0 ? stripped : stripped[..open];
        }

        // Remove complete <function_call>...</function_call> blocks
        int fcOpen = result.IndexOf("<function_call>", StringComparison.Ordinal);
        if (fcOpen >= 0)
        {
            result = System.Text.RegularExpressions.Regex.Replace(
                result, "<function_call>.*?</function_call>", string.Empty, System.Text.RegularExpressions.RegexOptions.Singleline);
            // Drop a trailing unclosed <function_call span
            int fcRemaining = result.IndexOf("<function_call", StringComparison.Ordinal);
            if (fcRemaining >= 0)
                result = result[..fcRemaining];
        }

        // Remove [TOOL_CALLS] marker (Mistral)
        result = result.Replace("[TOOL_CALLS]", string.Empty);

        // Remove <|python_tag|> marker (Llama-3) — and the pipe-less
        // <python_tag> variant a model may drift to.
        result = result.Replace("<|python_tag|>", string.Empty);
        result = result.Replace("<python_tag>", string.Empty);

        // Remove a leading "functionCall" marker (Gemma/Gemini-style)
        int fcMarker = result.IndexOf("functionCall", StringComparison.Ordinal);
        if (fcMarker >= 0 && fcMarker <= 1)
            result = result[(fcMarker + "functionCall".Length)..];

        return result;
    }

    /// <summary>
    /// True when the whole trimmed response is a JSON envelope in the agent's
    /// "Final Response Format" (<c>{ "status": "success"|"error", "data"|
    /// "message": ... }</c>) instead of an actual reply. A model that is still
    /// learning the tool contract emits this envelope rather than
    /// <c>&lt;tool_call&gt;</c> markup; showing it as prose just dumps JSON on
    /// the user. Only fires when there is no wrapper text at all, so a genuine
    /// answer that happens to quote one is untouched.
    /// </summary>
    private static bool IsFabricatedToolResultJson(string text)
    {
        string trimmed = text.Trim();
        if (trimmed.Length == 0 || trimmed[0] != '{') return false;
        try
        {
            if (JsonNode.Parse(trimmed) is not JsonObject obj) return false;
            if (obj["status"]?.GetValueKind() != JsonValueKind.String) return false;
            string status = obj["status"]!.GetValue<string>();
            return status is "success" or "error"
                && (obj.ContainsKey("data") || obj.ContainsKey("message"));
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>
    /// True when the response begins with a JSON object shaped like a tool call
    /// (<c>arguments</c> plus a <c>name</c>/<c>tool</c> key) that did not parse
    /// as one — truncated or malformed call machinery a small model emits when
    /// it runs out of attention (e.g. <c>{"name":"UIGetFreeMemory","arguments":{"}}</c>).
    /// Such output is neither a call to dispatch nor prose worth showing or
    /// persisting: drop it from both the stream and the history.
    /// </summary>
    private static bool LooksLikeFailedToolAttempt(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return false;
        int start = 0;
        while (start < text.Length && char.IsWhiteSpace(text[start])) start++;
        if (start >= text.Length || text[start] != '{') return false;
        string lower = text.ToLowerInvariant();
        return lower.Contains("\"arguments\"", StringComparison.Ordinal)
            && (lower.Contains("\"name\"", StringComparison.Ordinal)
                || lower.Contains("\"tool\"", StringComparison.Ordinal));
    }

    /// <summary>
    /// Returns the index just past the outermost JSON value that starts at
    /// <paramref name="start"/> (which must be <c>{</c> or <c>[</c>), honouring
    /// strings and escapes, or -1 when the value is not yet complete. Used to
    /// decide at stream time whether the response is still a pure JSON envelope
    /// that should be held back.
    /// </summary>
    private static int FindJsonValueEnd(string s, int start)
    {
        char open = s[start];
        char close = open == '{' ? '}' : open == '[' ? ']' : '\0';
        if (close == '\0') return -1;
        bool inString = false, escaped = false;
        int depth = 0;
        for (int i = start; i < s.Length; i++)
        {
            char c = s[i];
            if (inString)
            {
                if (escaped) { escaped = false; continue; }
                if (c == '\\') { escaped = true; continue; }
                if (c == '"') inString = false;
                continue;
            }
            if (c == '"') { inString = true; continue; }
            if (c == open) depth++;
            else if (c == close) { depth--; if (depth == 0) return i + 1; }
        }
        return -1;
    }

    /// <summary>
    /// Attempts to parse <paramref name="text"/> as a tool-call JSON object
    /// (with light repair: trailing comma removal, bracket truncation fixup).
    /// Returns true when the result is a <see cref="JsonObject"/> containing
    /// both a tool name and an <c>arguments</c> object. Accepts SharpMind's
    /// <c>tool</c> field or Qwen-instruct's native <c>name</c> field; the
    /// latter is normalized into <c>tool</c> so downstream code only reads one
    /// key.
    /// </summary>
    private static bool TryParseJsonObject(string text, out JsonObject? parsed)
    {
        parsed = null;

        // Light repair: remove trailing commas before ] or }
        string repaired = RegexGenerated.TrailingCommasBeforeClosingBracketOrBrace.Replace(text,"$1");// Regex.Replace(text, @",\s*([}\]])", "$1");

        // If the text ends without a closing brace and has an odd count of
        // opening vs closing braces, append the missing closing brace.
        int openBrace = repaired.Count(c => c == '{');
        int closeBrace = repaired.Count(c => c == '}');
        if (closeBrace < openBrace)
            repaired += new string('}', openBrace - closeBrace);

        try
        {
            var node = JsonNode.Parse(repaired);
            if (node is not JsonObject obj) return false;
            var nameNode = obj["tool"] ?? obj["name"];
            if (nameNode?.GetValueKind() != JsonValueKind.String) return false;
            if (obj["arguments"] is not JsonObject && obj["args"] is not JsonObject) return false;
            // Normalize the Qwen-native {"name": ...} shape so the rest of the
            // pipeline (dispatch, the ProcessToolRequest seam, the ToolCall
            // entry) always reads obj["tool"]. Remove "name" first — the node
            // still has "name" as its parent, so it cannot be re-attached under
            // "tool" while it sits there.
            if (obj["tool"] is null && obj["name"] is { } nameValue)
            {
                obj.Remove("name");
                obj["tool"] = nameValue;
            }
            // Normalize Gemma's {"args": ...} key to "arguments" so the rest of
            // the pipeline (line: toolCall["arguments"]!.AsObject()) works
            // unchanged.
            if (obj["args"] is { } argsValue)
            {
                obj.Remove("args");
                obj["arguments"] = argsValue;
            }
            parsed = obj;
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>
    /// Detects a failed native-JSON tool-call attempt: a complete JSON object
    /// with a top-level string "name" (or "tool") but no "arguments"/"args"
    /// block. Small native-format models (Qwen/Mistral/Llama3/Gemma) emit this
    /// when they reach for a tool but drop the arguments envelope; it must be
    /// suppressed as tool machinery, not surfaced as a JSON prose answer.
    /// </summary>
    private static bool IsFailedNativeCallAttempt(string text)
    {
        if (string.IsNullOrWhiteSpace(text) || text[0] != '{')
            return false;
        try
        {
            if (JsonNode.Parse(text) is not JsonObject obj) return false;
            var nameNode = obj["name"] ?? obj["tool"];
            if (nameNode?.GetValueKind() != JsonValueKind.String) return false;
            if (obj["arguments"] is JsonObject || obj["args"] is JsonObject) return false;
            return true;
        }
        catch (JsonException)
        {
            return false;
        }
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
            MaxNewTokens = MaxNewTokens,
            RepetitionPenalty = RepetitionPenalty,
            RepetitionWindow = RepetitionWindow,
            StopTokenIds = StopTokenIds ?? _tokenizer.GetEndOfGenerationIds(),
            StopStrings = MergeStopStrings(_formatter?.DefaultStopStrings, StopStrings),
            Stream = true,
        };

        var promptToks = _tokenizer.Encode(prompt, addBos: _addBos, addEos: false);

        _generator.ResetCache();
        // An unrelated prompt on the shared generator: the parent conversation's
        // cache is gone after this, and the next turn's prefix match rebuilds it.

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
    public void Interrupt() => _turnCts?.Cancel();

    public async IAsyncEnumerable<ChatStreamEntry> GetResponseStreamAsync(
        string userInput,
        ChatArtifact[]? artifacts = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        ThrowIfDisposed();
        EnsureInitialized();

        _history.Add(new ChatMessage { Role = ChatRole.User, Content = userInput, Name = _userName, Artifacts = artifacts });
        InvalidateHistoryCache();

        // Agentic loop: keep generating until the model produces a plain response
        // rather than a tool call, or until MaxToolCallsPerTurn is reached.
        for (int toolCallCount = 0; ; toolCallCount++)
        {
            // Tokenise the *full logical* prompt — what the whole conversation
            // renders to right now. `promptToks` is used for budget checks,
            // compaction, and next turn's bookkeeping.
            int[] promptToks = _tokenizer.Encode(BuildPrompt(), addBos: false, addEos: false);

            // Context compaction
            var compactor = _agentBuilder?.Compactor;
            if (compactor is not null)
            {
                var cmpCtx = new CompactionContext
                {
                    History = _history,
                    CurrentTokenCount = promptToks.Length,
                    MaxTokens = MaxTokens,
                    Model = _model,
                    Tokenizer = _tokenizer,
                    SummarizeAsync = async (text) =>
                    {
                        var sb = new System.Text.StringBuilder();
                        var sumCfg = new GenerationConfig { MaxNewTokens = 256, Stream = false };
                        var sumSmp = new SamplingConfig { Temperature = 0, TopK = 1 };
                        await foreach (var frag in _generator.GenerateAsync(text, sumSmp, sumCfg))
                            sb.Append(frag);
                        return sb.ToString();
                    }
                };
                if (await compactor.ShouldCompactAsync(cmpCtx, ct) && await compactor.CompactAsync(cmpCtx, ct))
                {
                    InvalidateHistoryCache();
                    promptToks = _tokenizer.Encode(BuildPrompt(), addBos: false, addEos: false);
                }
            }

            if (promptToks.Length > MaxTokens)
                promptToks = TrimToFitContext(promptToks);

            if (promptToks.Length == 0)
                throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");

            int[] generatorInput = FeedForPrompt(promptToks);

            var sampleCfg = new SamplingConfig
            {
                Temperature = Temperature,
                TopK = TopK,
                TopP = TopP,
            };

            var genCfg = new GenerationConfig
            {
                MaxNewTokens = MaxNewTokens,
                RepetitionPenalty = RepetitionPenalty,
                RepetitionWindow = RepetitionWindow,
                StopTokenIds = StopTokenIds ?? _tokenizer.GetEndOfGenerationIds(),
                StopStrings = MergeStopStrings(_formatter?.DefaultStopStrings, StopStrings),
                SlidingWindowSize = 0,
                Stream = true,
            };

            // Stream tokens
            _responseBuffer.Clear();
            List<ChatArtifact> responseArtifacts = [];

            // Streaming view of the visible text: `shownThrough` is how much has
            // been emitted, `droppedThrough` how much has been conclusively
            // classified as a fabricated tool-result envelope and suppressed.
            // A response that is still a pure JSON object is held back until it
            // resolves into either the envelope (dropped) or real prose (shown).
            int shownThrough = 0;
            int droppedThrough = 0;

            // When a native JSON tool-call object names a registered tool, the
            // whole turn is tool machinery: anything a small model emits after
            // the object is it answering before the result arrives (and would
            // duplicate what the tool itself will show). Set once, then the rest
            // of the turn is suppressed; the tool loop dispatches and regenerates.
            bool dropTurnTail = false;
            bool suppressedFailedCall = false;

            // Seed think-block detection from the rendered prompt, not just the
            // generated stream. Qwen3 and DeepSeek-R1 templates embed <think>\n
            // in the assistant prefix — the model never generates the opening tag,
            // so the stream-only check below would never fire.
            _inThinkBlock = PromptContainsUnclosedThinkTag();

            _turnCts?.Dispose();
            _turnCts = new CancellationTokenSource();
            using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(ct, _turnCts.Token);

            _progress?.Report(0f);
            int genFrags = 0;
            _prefillProgressQueue.Clear();
            LastPrefillTokenCount = generatorInput.Length;

            await foreach (var fragment in _generator.GenerateFromTokensAsync(generatorInput, sampleCfg, genCfg, linkedCts.Token))
            {
                // The generator prefills synchronously before yielding the first
                // fragment, so every progress callback for this turn is already
                // queued by now — surface it before the streamed text.
                foreach (var prefillEntry in DrainPrefillProgress())
                    yield return prefillEntry;

                _responseBuffer.Append(fragment);

                genFrags++;
                if (_progress is not null && genFrags % 5 == 0)
                    _progress.Report(Math.Min(genFrags / (float)genCfg.MaxNewTokens, 0.95f));

                // Safety: detect single-character repetition loop and stop
                if (_responseBuffer.Length >= 8)
                {
                    char last = _responseBuffer[^1];
                    bool loop = true;
                    for (int i = 2; i <= 8; i++)
                        if (_responseBuffer[^i] != last) { loop = false; break; }
                    if (loop) break;
                }

                // Detect <think> / </think> tag boundaries across fragments: the
                // block is open when the last opening tag comes after the last
                // closing one. Until the reply contains either tag the state seeded
                // from the prompt stands (Qwen3/DeepSeek-R1 templates open the block
                // in the assistant prefix). Checking only "contains <think>" here
                // used to re-arm on the already-closed tag, so every other token
                // after a think block was flagged as thinking.
{
                    string tail = _responseBuffer.ToString();
                    int lastOpen = tail.LastIndexOf("<think>", StringComparison.Ordinal);
                    int lastClose = tail.LastIndexOf("</think>", StringComparison.Ordinal);
                    if (lastOpen >= 0 || lastClose >= 0)
                        _inThinkBlock = lastOpen > lastClose;
                }

// Strip tag/tool markup from the whole accumulated buffer — never
                // from the fragment alone — so content split across fragments is
                // suppressed consistently and stray <tool_call> markup cannot leak
                // into the UI as prose.
                string visible = SanitizeToolMarkup(
                    _responseBuffer.ToString().Replace("<think>", "").Replace("</think>", ""));

                // Hold back a response that is still a pure JSON value on the
                // first pass: it might be the fabricated tool-result envelope,
                // which must not be shown. Once anything real commits (prose, a
                // non-envelope JSON answer) emission resumes normally.
                if (shownThrough == 0 && droppedThrough == 0 && visible.Length > 0)
                {
                    int start = 0;
                    while (start < visible.Length && char.IsWhiteSpace(visible[start])) start++;
                    if (start < visible.Length && (visible[start] == '{' || visible[start] == '['))
                    {
                        int valueEnd = FindJsonValueEnd(visible, start);
                        if (valueEnd < 0)
                        {
                            // The value is still open — it might yet close into
                            // the envelope, so hold everything back for now.
                            continue;
                        }
                        string jsonValue = visible[start..valueEnd];
                        if (string.IsNullOrWhiteSpace(visible[valueEnd..]) && IsFabricatedToolResultJson(jsonValue))
                        {
                            // Conclusively the envelope — drop it for good.
                            droppedThrough = valueEnd;
                        }
                        else if (_toolCallFormat != ToolCallFormat.None
                                 && TryParseJsonObject(jsonValue, out var heldCall))
                        {
                            // A leading call-shaped object ({"tool"/{"name"}+
                            // "arguments"}) is tool machinery, whether it is
                            // followed by prose (small models keep narrating after
                            // the object) or stands alone — never show it. It is
                            // either dispatched below or, if malformed, the
                            // stream still suppresses the attempt.
                            //
                            // Native JSON formats (Qwen/Mistral/Llama3/Gemma)
                            // contract the object alone: a registered tool call
                            // means the turn ends and the tool loop regenerates.
                            // Text after the object is the model answering before
                            // the result — suppress the whole tail so it cannot
                            // duplicate what the tool will show. SharpMind keeps
                            // the object-only drop so narration around tags stays.
                            string heldTool = heldCall?["tool"]?.GetValue<string>() ?? "";
                            if (_toolCallFormat != ToolCallFormat.SharpMind
                                && _agentBuilder is not null
                                && _agentBuilder.RegisteredToolNames.Contains(heldTool))
                            {
                                dropTurnTail = true;
                            }
                            else
                            {
                                droppedThrough = valueEnd;
                                // Small models often trail an extra stray '}'
                                // (sometimes several) after the object they just
                                // closed — swallow immediately-following closers
                                // and whitespace so machinery cannot leak as prose.
                                while (droppedThrough < visible.Length
                                       && (visible[droppedThrough] == '}' || visible[droppedThrough] == ']'
                                           || char.IsWhiteSpace(visible[droppedThrough])))
                                    droppedThrough++;
                            }
                        }
                        else if ((_toolCallFormat is ToolCallFormat.Qwen
                                  or ToolCallFormat.Mistral
                                  or ToolCallFormat.Llama3
                                  or ToolCallFormat.Gemma)
                                 && IsFailedNativeCallAttempt(jsonValue))
                        {
                            // A bare {"name":...} object with no "arguments" block
                            // is a failed tool-call attempt from a small
                            // native-JSON model (the qwen2-0.5B "Troll Troll"
                            // shape), not a prose answer. Suppress it; the turn
                            // then regenerates with a correction below.
                            droppedThrough = valueEnd;
                            while (droppedThrough < visible.Length
                                   && (visible[droppedThrough] == '}' || visible[droppedThrough] == ']'
                                       || char.IsWhiteSpace(visible[droppedThrough])))
                                droppedThrough++;
                            suppressedFailedCall = true;
                        }
                    }
                }

                if (dropTurnTail) continue;

                int emitStart = Math.Max(shownThrough, droppedThrough);
                if (emitStart >= visible.Length) continue;
                string delta = visible[emitStart..];
                shownThrough = visible.Length;

                // Always yield the real token content — the UI decides whether to
                // display vs suppress thinking tokens via ShowThinking.
                var ids = _generator.CurrentGeneratedIds;
                int? tokenId = ids != null && ids.Count > 0 ? ids[^1] : null;
                yield return new ChatStreamEntry
                {
                    Status = _inThinkBlock ? ChatStatus.Thinking : ChatStatus.Responding,
                    Token = delta,
                    IsComplete = false,
                    TokensPerSecond = _generator.TokensPerSecond,
                    TimeToFirstToken = _generator.TimeToFirstToken,
                    TokenId = tokenId,
                };
            }

            _progress?.Report(1f);
            var responseText = _responseBuffer.ToString();

            // Final call on a response held back as a possible JSON envelope that
            // never resolved while streaming: a genuine pure-JSON reply (or a
            // truncated JSON blob) still gets delivered; a fabricated envelope is
            // dropped below by the normal-response sanitization.
            if (!dropTurnTail && shownThrough == 0 && droppedThrough == 0 && responseText.Length > 0
                && !IsFabricatedToolResultJson(SanitizeToolMarkup(responseText))
                && !LooksLikeFailedToolAttempt(responseText))
            {
                string held = SanitizeToolMarkup(
                    responseText.Replace(" thinking", "").Replace("</think>", ""));
if (held.Length > 0)
                    yield return new ChatStreamEntry
                    {
                        Status = _inThinkBlock ? ChatStatus.Thinking : ChatStatus.Responding,
                        Token = held,
                        IsComplete = false,
                    };
            }

            // A suppressed failed tool-call attempt — a bare {"name":...} object
            // with no "arguments" block and nothing else shown — was pure tool
            // machinery that never became a call. Correct the model and regenerate
            // so the user receives plain prose instead of an empty turn.
            if (suppressedFailedCall && shownThrough == 0 && toolCallCount + 1 < MaxToolCallsPerTurn)
            {
                _history.Add(new ChatMessage
                {
                    Role = ResultRole,
                    Content = "Your previous reply was discarded: it contained a JSON object with a \"name\" field but no \"arguments\" block, which is not a valid tool call. Do not emit any JSON. Answer the user's request directly in plain prose, as if tools did not exist."
                });
                InvalidateHistoryCache();
                continue;
            }

            // Tool call detection. Guarded on RegisteredToolNames.Count > 0 and a known
            // call format so a session launched with tools disabled, an agent
            // builder with no registered tools, or an unknown model (None) can
            // never enter the tool-call loop, even if the model hallucinates a
            // <tool_call> tag or an envelope.
            if (_agentBuilder is not null
                && _agentBuilder.RegisteredToolNames.Count > 0
                && _toolCallFormat != ToolCallFormat.None
                && toolCallCount < MaxToolCallsPerTurn
                && TryParseToolCall(responseText, _toolCallFormat, out var toolCall)
                && toolCall is not null)
            {
                var toolName = toolCall["tool"]!.GetValue<string>();
                var args = toolCall["arguments"]!.AsObject();

                // Record the model's tool-call turn in history for the formatter
                _history.Add(ChatMessage.Agent(responseText, _agentBuilder?.AgentName));
                InvalidateHistoryCache();

                // Signal to the UI that a tool is about to execute
                yield return new ChatStreamEntry
                {
                    Status = ChatStatus.Executing,
                    Token = toolName,
                    IsComplete = false,
                    TokensPerSecond = _generator.TokensPerSecond,
                    TimeToFirstToken = _generator.TimeToFirstToken
                };

                // External interception seam. When set, the host decides what
                // happens to this call before the native agent dispatches it.
                if (ProcessToolRequest is not null)
                {
                    var external = await ProcessToolRequest(toolName, args, ct);
                    switch (external.Outcome)
                    {
                        case ToolRequestOutcome.Handled:
                            // The host ran the tool; feed its result back and
                            // continue the agentic loop as usual.
                            _history.Add(new ChatMessage
                            {
                                Role = ResultRole,
                                Content = $"Tool result: {external.Result}"
                            });
                            InvalidateHistoryCache();
                            continue;

                        case ToolRequestOutcome.ReturnToCaller:
                            // Hand the whole call back to the caller and end the
                            // turn without dispatching or feeding a result back.
                            yield return new ChatStreamEntry
                            {
                                Status = ChatStatus.ToolCall,
                                Token = toolName,
                                ToolCall = toolCall,
                                IsComplete = true,
                                TokensPerSecond = _generator.TokensPerSecond,
                                TimeToFirstToken = _generator.TimeToFirstToken
                            };
                            yield break;

                        case ToolRequestOutcome.Defer:
                        default:
                            // The host did not handle it; fall through to the
                            // native agent dispatch (with File/Network gating).
                            break;
                    }
                }

                // Dispatch with IO interception — interceptors gate any actual
                // file/network access the tool makes through PermissionCallback.
                // If PermissionCallback is null, interceptors are never activated.
                var toolResult = await DispatchToolAsync(toolName, toolCall, args, ct);

                // Feed the result back as a system message for the next generation pass
                _history.Add(new ChatMessage
                {
                    Role = ResultRole,
                    Content = $"Tool result: {toolResult.ToJsonString()}"
                });
                InvalidateHistoryCache();
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
                InvalidateHistoryCache();

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
                    Role = ResultRole,
                    Content = $"Tool result: {agentResult}"
                });
                InvalidateHistoryCache();
                continue;                   // generate again with enriched history
            }

            // Depth limit reached — inform the model
            if (_agentBuilder is not null
                && _agentBuilder.AgentsEnabled
                && toolCallCount < MaxToolCallsPerTurn
                && _currentDepth >= MaxAgentDepth
                && TryParseAgentTag(responseText, out _, out _, out _, out _))
            {
                _history.Add(ChatMessage.Agent(responseText, _agentBuilder?.AgentName));
                _history.Add(new ChatMessage
                {
                    Role = ChatRole.System,
                    Content = $"Tool result: {{\"status\":\"error\",\"message\":\"Maximum agent depth ({MaxAgentDepth}) reached. Cannot delegate further.\"}}"
                });
                InvalidateHistoryCache();
                continue;
            }

            // Normal (non-tool) response — sanitized so tool machinery that never
            // became a call (stray <tool_call> markup, or a response that is only
            // the fabricated tool-result envelope) is not persisted as prose.
            responseText = SanitizeToolMarkup(responseText);
            if (IsFabricatedToolResultJson(responseText))
                responseText = string.Empty;
            if (LooksLikeFailedToolAttempt(responseText))
                responseText = string.Empty;
            if (responseText.Length > 0)
            {
                var agentMsg = ChatMessage.Agent(responseText, _agentBuilder?.AgentName);
                if (responseArtifacts.Count > 0)
                    agentMsg.Artifacts = [.. responseArtifacts];
                _history.Add(agentMsg);
                InvalidateHistoryCache();
            }

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
        EnsureInitialized();
        while (!token.IsCancellationRequested)
        {
            response(new ChatStreamEntry { Status = ChatStatus.Waiting, IsComplete = false });
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

            // Pre-process user input (pre-processor can modify input.Content, input.Artifacts, etc. in place)
            if (_preProcessor is not null)
                await _preProcessor.ProcessAsync(input, _history, token);

            try
            {
                await foreach (var entry in GetResponseStreamAsync(input.Content, input.Artifacts, token))
                {
                    response(entry);
                    TokensPerSecond = entry.TokensPerSecond;
                    TimeToFirstToken = entry.TimeToFirstToken;
                }

                if (_postProcessor is not null && _history.Count > 0 && _history[^1].Role == ChatRole.Agent)
                    await _postProcessor.ProcessAsync(_history[^1], _history, token);
            }
            catch (OperationCanceledException)
            {
                if (_responseBuffer.Length > 0)
                    _pendingDraft = _responseBuffer.ToString();
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted, IsComplete = true, TokensPerSecond = _generator.TokensPerSecond, TimeToFirstToken = _generator.TimeToFirstToken });
                if (token.IsCancellationRequested) break;
                continue;
            }
            catch (Exception ex)
            {
                // Still caught so a host UI is not torn down by a bad turn, but the
                // reason travels with the entry now: swallowing it made an internal
                // failure look identical to an empty answer.
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted, IsComplete = true, TokensPerSecond = _generator.TokensPerSecond, TimeToFirstToken = _generator.TimeToFirstToken, Error = $"{ex.GetType().Name}: {ex.Message}" });
                break;
            }

        }
        return [.. _history];
    }

    /// <summary>
    /// Converts queued prefill-progress fractions into stream entries shown in
    /// the host UI's status line ("Prefilling 50.25%"). Returns nothing when the
    /// prompt was short enough to prefill in a single chunk (no progress events
    /// were reported) or when the KV cache was extended incrementally.
    /// </summary>
    private IEnumerable<ChatStreamEntry> DrainPrefillProgress()
    {
        while (_prefillProgressQueue.Count > 0)
        {
            double fraction = _prefillProgressQueue.Dequeue();
            yield return new ChatStreamEntry
            {
                Status = ChatStatus.Updating,
                Token = $"Prefilling {fraction * 100:F2}%",
                IsComplete = false,
            };
        }
    }

    public async Task<ChatMessage[]> StartChatAsync(Func<ChatMessage> prompt, Action<ChatStreamEntry> response, CancellationToken token = default)
        => await StartChatAsync(() => Task.FromResult(prompt()), response, token);

    public async Task<ChatMessage[]> StartChatAsync(Func<Task<string>> prompt, Action<string> response, CancellationToken token = default)
        => await StartChatAsync(async () => new ChatMessage { Content = await prompt(), Role = ChatRole.User, Name = _userName }, (e) =>
        {
            if (e.Token is { Length: > 0 } delta) response(delta);
        }, token);

    public async Task<ChatMessage[]> StartChatAsync(Func<string> prompt, Action<string> response, CancellationToken token = default)
        => await StartChatAsync(() => new ChatMessage { Content = prompt(), Role = ChatRole.User, Name = _userName }, (e) =>
        {
            if (e.Token is { Length: > 0 } delta) response(delta);
        }, token);

    public ChatSessionSnapshot GetSnapshot()
    {
        ThrowIfDisposed();
        KVCacheSnapshot? kvCache = null;
        var cacheTokens = _generator?.CacheTokens;
        if (_hasEngine && _engine is not null && cacheTokens is not null && cacheTokens.Count > 0)
        {
            // Engine-backed caches have no per-layer host blob (EngineKVCacheView throws);
            // the whole-engine snapshot is the engine's own responsibility.
            kvCache = _engine.ExportCache([.. cacheTokens]);
        }
        else if (ActiveCaches is { Length: > 0 } ac && cacheTokens is not null && cacheTokens.Count > 0)
        {
            var layers = new List<byte[]>();
            bool anyData = false;
            foreach (var cache in ac)
            {
                var blob = cache.SnapshotBytes();
                if (blob is not null) { layers.Add(blob); anyData = true; }
                else layers.Add([]);
            }
            if (anyData)
                kvCache = new KVCacheSnapshot
                {
                    PromptHash = KVCacheSnapshot.HashPromptTokens([.. cacheTokens]),
                    PromptTokenCount = cacheTokens.Count,
                    Layers = layers
                };
        }
        return new ChatSessionSnapshot
        {
            History = [.. _history],
            PendingDraft = _pendingDraft,
            KVCache = kvCache
        };
    }

    public void LoadSnapshot(ChatSessionSnapshot snapshot)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(snapshot);

        _history.Clear();
        _pendingDraft = snapshot.PendingDraft;

        if (_agentBuilder != null)
        {
            foreach (string prompt in _agentBuilder.AdditionalSystemPrompts)
                _history.Add(new ChatMessage { Role = ChatRole.System, Content = prompt });
            _history.Add(new ChatMessage { Role = ChatRole.System, Content = _agentBuilder.BuildAgentPrompt() });
        }

        foreach (var msg in snapshot.History)
            if (msg.Role != ChatRole.System)
                _history.Add(msg);
        InvalidateHistoryCache();
        _generator.ResetCache();

        // Restore KV-cache state if the snapshot carried one.
        // The hash check in WarmupPrefillAsync decides whether to use it.
        _pendingKvCacheSnapshot = snapshot.KVCache;
    }

    public static readonly JsonSerializerOptions IndentedEnumConverter = new()
    {
        WriteIndented = true,
        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() }
    };
    public static readonly JsonSerializerOptions EnumConverter = new()
    {
        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() }
    };
    public async Task SaveAsync(string path, CancellationToken ct = default)
    {
        ThrowIfDisposed();
        var snapshot = GetSnapshot();
        var json = JsonSerializer.Serialize(snapshot, IndentedEnumConverter);
        await File.WriteAllTextAsync(path, json, ct);
    }

    public static async Task LoadSnapshotAsync(string path, ChatSession<T, K> session, CancellationToken ct = default)
    {
        var json = await File.ReadAllTextAsync(path, ct);
        var snapshot = JsonSerializer.Deserialize<ChatSessionSnapshot>(json, EnumConverter)
            ?? throw new InvalidOperationException("Failed to deserialize session snapshot.");
        session.LoadSnapshot(snapshot);
    }
}
