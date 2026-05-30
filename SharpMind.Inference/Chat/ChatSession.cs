using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.CompilerServices;

namespace SharpMind.Inference.Chat;

public sealed class ChatSession
{
    private readonly Transformer _model;
    private readonly Tokenizer _tokenizer;
    private readonly Generator _generator;
    private readonly List<ChatMessage> _history = [];
    private readonly IChatPromptFormatter? _formatter;
    private readonly bool _addBos;
    private bool _disposed;

    public ChatSession(
        Transformer model,
        Tokenizer tokenizer,
        InferenceOps ops,
        GgufMeta? meta = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(ops);

        _model = model;
        _tokenizer = tokenizer;
        _generator = new Generator(model, tokenizer, ops);
        _formatter = ChatPromptFormatterFactory.Create(meta);//.GetChatTemplate(), meta?.GetString("general.name"));
        _addBos = meta?.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
    }

    public Transformer Model => _model;
    public Tokenizer Tokenizer => _tokenizer;
    public Generator Generator => _generator;
    public IReadOnlyList<ChatMessage> History => _history;

    public int MaxTokens { get; set; } = 2048;
    public float Temperature { get; set; } = 0.0f;
    public int TopK { get; set; } = 20;
    public float TopP { get; set; } = 0.85f;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public int RepetitionWindow { get; set; } = 32;
    public float? TokensPerSecond { get; private set; }

    public async IAsyncEnumerable<ChatStreamEntry> GetResponseStreamAsync(
        string userInput,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        await Task.CompletedTask;
        ThrowIfDisposed();

        yield return new ChatStreamEntry { Status = ChatStatus.Thinking };

        var userMessage = ChatMessage.User(userInput);
        _history.Add(userMessage);

        yield return new ChatStreamEntry { Status = ChatStatus.Updating };
        yield return new ChatStreamEntry { Status = ChatStatus.Responding, IsComplete = false };

        var prompt = BuildPrompt();
        int[] promptToks = _tokenizer.Encode(prompt, addBos: false, addEos: false);
        if (promptToks.Length > MaxTokens)
        {
            int start = promptToks.Length - MaxTokens;
            var subset = new int[MaxTokens];
            Array.Copy(promptToks, start, subset, 0, MaxTokens);
            promptToks = subset;
        }

        if (promptToks.Length == 0)
            throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");

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

            // Safety: detect repetition loop and stop
            if (response.Length >= 8)
            {
                int len = response.Length;
                bool loop = true;
                for (int i = 0; i < 4; i++)
                    if (response[len - 4 + i] != response[len - 8 + i]) { loop = false; break; }
                if (loop) break;
            }

            yield return new ChatStreamEntry
            {
                Status = ChatStatus.Responding,
                TextDelta = fragment,
                IsComplete = false,
                TokensPerSecond = _generator.TokensPerSecond
            };
        }

        if (response.Length > 0)
            _history.Add(ChatMessage.Agent(response.ToString()));

        yield return new ChatStreamEntry
        {
            Status = ChatStatus.Complete,
            IsComplete = true,
            TokensPerSecond = _generator.CumulativeTokensPerSecond
        };
    }

    public async Task<ChatResult> GetResponseAsync(
        string userInput,
        CancellationToken ct = default)
    {
        var entries = new List<ChatStreamEntry>();
        var content = new System.Text.StringBuilder();

        await foreach (var entry in GetResponseStreamAsync(userInput, ct))
        {
            entries.Add(entry);
            if (entry.TextDelta is not null)
                content.Append(entry.TextDelta);
        }

        var lastEntry = entries.LastOrDefault();

        return new ChatResult
        {
            Status = lastEntry?.Status ?? ChatStatus.Complete,
            Content = content.ToString(),
            IsComplete = true,
            IsStreaming = true
        };
    }

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
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(_disposed, nameof(ChatSession));

    //Review: do hide system/agent prompts? may change over time not revelent to actual chat history
    public async Task<ChatMessage[]> StartChatAsync(CancellationToken token, Func<ChatMessage> prompt, Action<ChatStreamEntry> response)
    {
        while (!token.IsCancellationRequested)
        {
            var input = prompt();
            if (string.IsNullOrWhiteSpace(input.Content))
                continue;
            try
            {
                await foreach (var entry in GetResponseStreamAsync(input.Content, token))
                {
                    if (entry.TextDelta is { Length: > 0 } delta) response(entry);
                    TokensPerSecond = entry.TokensPerSecond;
                }
            }
            catch (OperationCanceledException)
            {
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted });
            }
            catch
            {
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted });
            }
        }
        return [.. _history];
    }
    public async Task<ChatMessage[]> StartChatAsync(CancellationToken token, Func<string> prompt, Action<string> response)
    {
        return await StartChatAsync(token, () => new ChatMessage { Content = prompt(), Role = ChatRole.User }, (e) =>
        {
            if (e.TextDelta is { Length: > 0 } delta) response(delta);
        });
    }
}
