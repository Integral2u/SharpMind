using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;

namespace SharpMind.Inference.Chat;

public sealed class ChatSession
{
    private readonly Tokenizer _tokenizer;
    private readonly IGenerator _generator;
    private readonly List<ChatMessage> _history = [];
    private readonly IChatPromptFormatter? _formatter;
    private readonly bool _addBos;
    private bool _disposed;

    public ChatSession(
        IGenerator generator,
        Tokenizer tokenizer,
        GgufMeta? meta = null)
    {
        ArgumentNullException.ThrowIfNull(generator);
        ArgumentNullException.ThrowIfNull(tokenizer);

        _generator = generator;
        _tokenizer = tokenizer;
        _formatter = ChatPromptFormatterFactory.Create(meta);
        _addBos = meta?.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
    }

    public Tokenizer Tokenizer => _tokenizer;
    public IReadOnlyList<ChatMessage> History => _history;

    public int MaxTokens { get; set; } = 2048;
    public float Temperature { get; set; } = 0.0f;
    public int TopK { get; set; } = 20;
    public float TopP { get; set; } = 0.85f;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public int RepetitionWindow { get; set; } = 32;
    public float? TokensPerSecond { get; private set; }


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

    private async IAsyncEnumerable<ChatStreamEntry> GetResponseStreamAsync(
    string userInput,
    [EnumeratorCancellation] CancellationToken ct = default)
    {
        ThrowIfDisposed();

        var userMessage = ChatMessage.User(userInput);
        _history.Add(userMessage);

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
                }
            }
            catch (OperationCanceledException)
            {
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted, IsComplete = true , TokensPerSecond = _generator.CumulativeTokensPerSecond });
                break;
            }
            catch
            {
                response(new ChatStreamEntry { Status = ChatStatus.Interrupted, IsComplete = true, TokensPerSecond = _generator.CumulativeTokensPerSecond });
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
