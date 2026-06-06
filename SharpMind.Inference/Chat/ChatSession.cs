using SharpMind.Inference.Agent;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.CompilerServices;

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

    private async IAsyncEnumerable<ChatStreamEntry> GetResponseStreamAsync(
    string userInput,
    [EnumeratorCancellation] CancellationToken ct = default)
    {
        ThrowIfDisposed();

        var userMessage = ChatMessage.User(userInput);
        _history.Add(userMessage);

        int[] promptToks;
        if (_cachedPromptTokens is not null && _formatter is null)
        {
            // Incremental: previous prompt ended with "assistant: ".
            // Append <response>\nuser: <input>\nassistant:  to form the new prompt.
            var incremental = new System.Text.StringBuilder();
            if (_history.Count >= 2 && _history[^2].Role == ChatRole.Agent)
                incremental.Append(_history[^2].Content).Append('\n');
            incremental.Append("user: ").Append(userInput).Append("\nassistant: ");

            int[] newTokens = _tokenizer.Encode(incremental.ToString(), addBos: false, addEos: false);
            promptToks = GC.AllocateUninitializedArray<int>(_cachedPromptTokens.Length + newTokens.Length);
            _cachedPromptTokens.CopyTo(promptToks.AsSpan());
            newTokens.CopyTo(promptToks.AsSpan(_cachedPromptTokens.Length));
        }
        else
        {
            // Full re-encode (first turn or formatter in use)
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

        var response = new System.Text.StringBuilder();

        await foreach (var fragment in _generator.GenerateFromTokensAsync(promptToks, sampleCfg, genCfg, ct))
        {
            response.Append(fragment);

            // Safety: detect single-character repetition loop and stop
            if (response.Length >= 8)
            {
                char lastChar = response[^1];
                bool loop = true;
                for (int i = 2; i <= 8; i++)
                    if (response[^i] != lastChar) { loop = false; break; }
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

        if (response.Length > 0)
            _history.Add(ChatMessage.Agent(response.ToString()));

        yield return new ChatStreamEntry
        {
            Status = ChatStatus.Complete,
            IsComplete = true,
            TokensPerSecond = _generator.TokensPerSecond,
            TimeToFirstToken = _generator.TimeToFirstToken
        };
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
