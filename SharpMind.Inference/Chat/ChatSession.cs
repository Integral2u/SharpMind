using SharpMind.Core.Tensors;
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
    private readonly InferenceOps _ops;
    private readonly KVCache[] _caches;
    private readonly List<ChatMessage> _history = [];
    private readonly int[] _decodeTokenScratch = new int[1];
    private readonly IChatPromptFormatter? _formatter;
    private readonly bool _addBos;
    private bool _disposed;

    /// <summary>Legend delimiter used in debug output to separate prompt from generation.</summary>
    private const string DebugLegend = "\n│ Prompt │\n";

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
        _ops = ops;
        _formatter = ChatPromptFormatterFactory.Create(meta?.GetChatTemplate());
        _addBos = meta?.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;

        int nl = model.Config.NumLayers;
        int ms = model.Config.MaxSeqLen;
        int nk = model.Config.NumKvHeads;
        int hd = model.Config.HeadDim;

        _caches = new KVCache[nl];
        for (int i = 0; i < nl; i++)
            _caches[i] = new KVCache(1, nk, ms, hd);
    }

    public Transformer Model => _model;
    public Tokenizer Tokenizer => _tokenizer;
    public InferenceOps Ops => _ops;
    public IReadOnlyList<ChatMessage> History => _history;

    public int MaxTokens { get; set; } = 2048;
    public float Temperature { get; set; } = 0.0f;
    public int TopK { get; set; } = 20;
    public float TopP { get; set; } = 0.85f;
    public float RepetitionPenalty { get; set; } = 1.1f;
    public int RepetitionWindow { get; set; } = 32;

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
        var encoded = _tokenizer.Encode(prompt, addBos: false, addEos: false);
        int[] promptToks;
        if (encoded.Length > MaxTokens)
        {
            promptToks = encoded.ToArray();
            ArraySegment<int> subset = new(encoded.ToArray(), encoded.Length - MaxTokens, MaxTokens);
            promptToks = subset.ToArray();
        }
        else
        {
            promptToks = encoded.ToArray();
        }

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

        // ForwardLastLogits returns [batch, vocabSize] � just the last token's logits.
        // This avoids allocating and slicing the full [batch, seqLen, vocab] tensor
        // that Forward() returns, and removes the step==0 / step>0 branch below.
        Tensor<float>? logitsTensor = _model.ForwardLastLogits(input, _caches, posOffset);

        try
        {
            int vocabSize = logitsTensor.Shape[1];

            var samplingCfg = new SamplingConfig
            {
                Temperature = Temperature,
                TopK = TopK,
                TopP = TopP
            };

            var generatedIds = new List<int>();
            var response = new System.Text.StringBuilder();

            for (int step = 0; step < MaxTokens; step++)
            {
                if (ct.IsCancellationRequested) break;

                // logitsTensor is always [1, vocabSize] — same layout on every step.
                ReadOnlySpan<float> logitsSlice = logitsTensor.Data[..vocabSize];

                float[]? logitsCopy = null;
                Span<float> logitsSpan;
                if (RepetitionPenalty != 1.0f && generatedIds.Count > 0)
                {
                    logitsCopy = System.Buffers.ArrayPool<float>.Shared.Rent(vocabSize);
                    logitsSlice.CopyTo(logitsCopy);
                    logitsSpan = logitsCopy.AsSpan(0, vocabSize);
                    ApplyRepetitionPenalty(logitsSpan, promptToks, generatedIds, RepetitionPenalty, RepetitionWindow);
                }
                else
                {
                    logitsSpan = logitsSlice.ToArray().AsSpan();
                }

                int nextId = Sampler.Sample(logitsSpan, samplingCfg, Random.Shared);
                generatedIds.Add(nextId);

                if (nextId == _tokenizer.EosId) break;

                // Safety: detect repetition loop and stop
                if (step > 4)
                {
                    string last4 = response.ToString().Substring(response.Length - 4);
                    if (response.Length >= 8)
                    {
                        string last8 = response.ToString().Substring(response.Length - 8);
                        if (last8 == last4 + last4)
                        {
                            // Detected repetition loop
                            break;
                        }
                    }
                }

                _decodeTokenScratch[0] = nextId;
                var token = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1));

                response.Append(token);

                yield return new ChatStreamEntry
                {
                    Status = ChatStatus.Responding,
                    TextDelta = token,
                    IsComplete = false
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
    public void AddMessage(ChatMessage message)
    {
        ThrowIfDisposed();
        _history.Add(message);
    }
    /// <summary>
    /// Returns the formatted prompt string that would be sent to the model
    /// for the current history. Useful for debugging and comparing against
    /// reference implementations (LLamaSharp, llama.cpp).
    /// </summary>
    public string GetFormattedPrompt()
    {
        ThrowIfDisposed();
        return BuildPrompt();
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
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(_disposed, nameof(ChatSession));

    private static void ApplyRepetitionPenalty(
        Span<float> logits,
        ReadOnlySpan<int> promptIds,
        List<int> generatedIds,
        float penalty,
        int window)
    {
        static void ScaleId(Span<float> lg, int id, float pen)
        {
            if ((uint)id >= (uint)lg.Length) return;
            lg[id] = lg[id] >= 0f ? lg[id] / pen : lg[id] * pen;
        }

        if (window > 0)
        {
            int start = Math.Max(0, generatedIds.Count - window);
            for (int i = start; i < generatedIds.Count; i++)
                ScaleId(logits, generatedIds[i], penalty);
            return;
        }

        foreach (int id in promptIds)
            ScaleId(logits, id, penalty);
        for (int i = 0; i < generatedIds.Count; i++)
            ScaleId(logits, generatedIds[i], penalty);
    }

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

                }
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