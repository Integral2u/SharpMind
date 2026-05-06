using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Tokenization;

namespace SharpMind.Inference.Chat;

/// <summary>
/// Chat roles for conversation participants.
/// </summary>
public enum ChatRole
{
    /// <summary>System prompt - sets behavior/instructions.</summary>
    System,
    /// <summary>AI assistant/agent responses.</summary>
    Agent,
    /// <summary>Human user input.</summary>
    User
}

/// <summary>
/// Status values during chat response generation.
/// </summary>
public enum ChatStatus
{
    /// <summary>Analyzing request, planning response.</summary>
    Thinking,
    /// <summary>Updating context/history.</summary>
    Updating,
    /// <summary>Executing tools/skills.</summary>
    Executing,
    /// <summary>Generating text response.</summary>
    Responding,
    /// <summary>Waiting for input or tool results.</summary>
    Waiting,
    /// <summary>Completed.</summary>
    Complete
}

/// <summary>
/// A single message in the chat conversation.
/// </summary>
public sealed class ChatMessage
{
    public required ChatRole Role { get; init; }
    public required string Content { get; init; }
    public string? Name { get; init; }
    public DateTime Timestamp { get; init; } = DateTime.UtcNow;

    public static ChatMessage User(string content)
        => new() { Role = ChatRole.User, Content = content };

    public static ChatMessage System(string content)
        => new() { Role = ChatRole.System, Content = content };

    public static ChatMessage Agent(string content, string? name = null)
        => new() { Role = ChatRole.Agent, Content = content, Name = name };
}

/// <summary>
/// Result from chat response generation.
/// Can be streamed as it's being generated.
/// </summary>
public sealed class ChatResult
{
    public ChatStatus Status { get; internal init; }
    public string? Content { get; internal init; }
    public List<ChatArtifact>? Artifacts { get; internal init; }
    public bool IsStreaming { get; internal init; }
    public bool IsComplete { get; internal init; }
    public string? Error { get; internal init; }
}

/// <summary>
/// Artifact attached to a chat response (images, code blocks, etc.).
/// </summary>
public sealed class ChatArtifact
{
    public required string Type { get; init; }  // "text", "image", "code", "json"
    public required string Content { get; init; }
    public string? Language { get; init; }
    public string? FileName { get; init; }
}

/// <summary>
/// Streaming response entry for real-time updates.
/// </summary>
public sealed class ChatStreamEntry
{
    public required ChatStatus Status { get; init; }
    public string? TextDelta { get; init; }
    public ChatArtifact? Artifact { get; init; }
    public bool IsComplete { get; init; }
}

/// <summary>
/// Main chat session - handles conversation with model.
/// </summary>
public sealed class ChatSession : IAsyncDisposable
{
    private readonly Transformer _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly InferenceOps _ops;
    private readonly KVCache[] _caches;
    private readonly List<ChatMessage> _history = [];
    private bool _disposed;

    public ChatSession(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        InferenceOps ops)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(ops);

        _model = model;
        _tokenizer = tokenizer;
        _ops = ops;

        int nl = model.Config.NumLayers;
        int ms = model.Config.MaxSeqLen;
        int nk = model.Config.NumKvHeads;
        int hd = model.Config.HeadDim;

        _caches = new KVCache[nl];
        for (int i = 0; i < nl; i++)
            _caches[i] = new KVCache(1, nk, ms, hd);
    }

    public Transformer Model => _model;
    public Tokenization.Tokenizer Tokenizer => _tokenizer;
    public InferenceOps Ops => _ops;
    public IReadOnlyList<ChatMessage> History => _history;

    public int MaxTokens { get; set; } = 2048;
    public float Temperature { get; set; } = 0.7f;
    public int TopK { get; set; } = 40;
    public float TopP { get; set; } = 0.9f;

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
        var promptIds = _tokenizer.Encode(prompt, addBos: true, addEos: false);
        var posOffset = _caches[0].Length;

        using var input = Tensor<int>.From(promptIds, 1, promptIds.Length);

        if (promptIds.Length > MaxTokens)
        {
            promptIds = promptIds.TakeLast(MaxTokens).ToArray();
            input.Dispose();
        }

        using var logits = _model.Forward(input, _caches, posOffset);
        var vocabSize = logits.Shape.Cols;

        int generated = 0;
        var response = new System.Text.StringBuilder();

        while (generated < MaxTokens)
        {
            if (ct.IsCancellationRequested) break;

            var lastLogits = logits.Data.Slice((promptIds.Length - 1) * vocabSize, vocabSize);
            var nextId = Sample(lastLogits.ToArray());

            if (nextId == _tokenizer.EosId) break;

            var token = _tokenizer.Decode([nextId]);
            response.Append(token);

            var chatToken = ChatMessage.Agent(token);
            _history.Add(chatToken);

            yield return new ChatStreamEntry
            {
                Status = ChatStatus.Responding,
                TextDelta = token,
                IsComplete = false
            };

            using var nextInput = Tensor<int>.From([nextId], 1, 1);
            int newPos = posOffset + promptIds.Length + generated + 1;
            using var nextLogits = _model.Forward(nextInput, _caches, newPos);

            promptIds = [.. promptIds, nextId];
            generated++;
        }

        yield return new ChatStreamEntry { Status = ChatStatus.Complete, IsComplete = true };
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

    public void ClearHistory()
    {
        _history.Clear();
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Reset();
    }

    public void ResetCaches()
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Reset();
    }

    private string BuildPrompt()
    {
        var sb = new System.Text.StringBuilder();

        foreach (var msg in _history)
        {
            var prefix = msg.Role switch
            {
                ChatRole.System => "system: ",
                ChatRole.Agent => "assistant: ",
                ChatRole.User => "user: ",
                _ => ""
            };
            sb.AppendLine(prefix + msg.Content);
        }

        return sb.ToString();
    }

    private int Sample(float[] logits)
    {
        var cfg = new SamplingConfig
        {
            Temperature = Temperature,
            TopK = TopK,
            TopP = TopP
        };

        return Sampler.Sample(logits, cfg, Random.Shared);
    }

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        await Task.CompletedTask;
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(_disposed, nameof(ChatSession));
}