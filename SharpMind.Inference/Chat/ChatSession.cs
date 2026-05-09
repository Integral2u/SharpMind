using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
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
    /// <summary>That chat was interrupted, cancelled or failed.</summary>
    Interrupted,
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
    /// <summary>Reused single-token decode buffer (avoids a new int[] each step).</summary>
    private readonly int[] _decodeTokenScratch = new int[1];
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
        var encoded = _tokenizer.Encode(prompt, addBos: true, addEos: false);
        ReadOnlySpan<int> promptToks = encoded;
        if (encoded.Length > MaxTokens)
            promptToks = encoded.AsSpan(encoded.Length - MaxTokens);

        int promptLen = promptToks.Length;
        if (promptLen == 0)
            throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");

        // BuildPrompt() always re-encodes the entire conversation history from scratch.
        // Re-using a stale KV cache from a previous turn would write into already-occupied
        // positions (posOffset = old cache length, but the full prompt is passed again),
        // corrupting both the cache content and position encodings.
        // Resetting here is correct: the full prompt encodes all context the model needs.
        ResetCaches();
        var posOffset = 0;

        using var input = Tensor<int>.From(promptToks, 1, promptLen);

        // ForwardLastLogits returns [batch, vocabSize] — just the last token's logits.
        // This avoids allocating and slicing the full [batch, seqLen, vocab] tensor
        // that Forward() returns, and removes the step==0 / step>0 branch below.
        Tensor<float>? logitsTensor = _model.ForwardLastLogits(input, _caches, posOffset);

        try
        {
            int vocabSize = logitsTensor.Shape[1];
            var response = new System.Text.StringBuilder();

            var samplingCfg = new SamplingConfig
            {
                Temperature = Temperature,
                TopK        = TopK,
                TopP        = TopP
            };

            for (int step = 0; step < MaxTokens; step++)
            {
                if (ct.IsCancellationRequested) break;

                // logitsTensor is always [1, vocabSize] — same layout on every step.
                ReadOnlySpan<float> logitsSlice = logitsTensor.Data[..vocabSize];

                int nextId = Sampler.Sample(logitsSlice, samplingCfg, Random.Shared);

                if (nextId == _tokenizer.EosId) break;

                _decodeTokenScratch[0] = nextId;
                var token = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1));

                response.Append(token);

                yield return new ChatStreamEntry
                {
                    Status      = ChatStatus.Responding,
                    TextDelta   = token,
                    IsComplete  = false
                };

                Tensor<float>? prev = logitsTensor;
                logitsTensor = null;
                int newPos = posOffset + promptLen + step;

                prev.Dispose();
                using var nextInput = Tensor<int>.From(_decodeTokenScratch.AsSpan(0, 1), 1, 1);
                logitsTensor = _model.ForwardLastLogits(nextInput, _caches, newPos);
            }

            // One consolidated agent message avoids O(tokens) retained chat rows and prompt blow-ups.
            if (response.Length > 0)
                _history.Add(ChatMessage.Agent(response.ToString()));
        }
        finally
        {
            logitsTensor?.Dispose();
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

        // Prime the model to generate an assistant turn.
        // Without this cue the model sees "user: <input>\n" as an incomplete
        // user utterance and echoes/continues user text instead of replying.
        sb.Append("assistant: ");

        return sb.ToString();
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