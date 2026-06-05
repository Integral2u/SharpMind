using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <summary>
/// Token generation loop for autoregressive decoding.
/// </summary>
public sealed class StandardGenerator : IGenerator
{
    private readonly Transformer  _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly IKVCache[]     _caches;
    private readonly Random       _defaultRng;
    /// <summary>Reused decode step (<c>[1]</c>) to avoid allocating a new <see cref="int"/>[] each token.</summary>
    private readonly int[]       _decodeTokenScratch = new int[1];
    /// <summary>Cached scratch buffer for repetition-penalty copy to avoid <see cref="ArrayPool{T}.Rent"/> per token.</summary>
    private float[]?              _penaltyScratch;
    private bool                  _disposed;

    public StandardGenerator(
        Transformer   model,
        Tokenization.Tokenizer tokenizer,
        IKVCache[]?   caches = null,
        int?          seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        _model      = model;
        _tokenizer  = tokenizer;

        if (caches != null)
        {
            _caches = caches;
        }
        else
        {
            int numLayers = model.Config.NumLayers;
            int maxSeqLen = model.Config.MaxSeqLen;
            int numKvHeads = model.Config.NumKvHeads;
            int headDim   = model.Config.HeadDim;

            _caches = new IKVCache[numLayers];
            for (int i = 0; i < numLayers; i++)
                _caches[i] = new KVCache(1, numKvHeads, maxSeqLen, headDim);
        }

        _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>
    /// Generates tokens from <paramref name="prompt"/> as an async stream of
    /// decoded text fragments.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateAsync(
        string                                    prompt,
        SamplingConfig?                           sampling   = null,
        GenerationConfig?                         generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        int[] promptIds = _tokenizer.Encode(prompt, addBos: true, addEos: false);
        await foreach (var fragment in GenerateFromTokensAsync(promptIds, sampling, generation, cancellationToken))
            yield return fragment;
    }

    /// <summary>
    /// Generates tokens from already-encoded token IDs.
    /// The caller is responsible for including any desired BOS token in the array.
    /// </summary>
    public async IAsyncEnumerable<string> GenerateFromTokensAsync(
        int[]                                     promptIds,
        SamplingConfig?                           sampling   = null,
        GenerationConfig?                         generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        ThrowIfDisposed();

        var sampleCfg = sampling   ?? SamplingConfig.Greedy;
        var genCfg    = generation ?? GenerationConfig.Default;

        if (promptIds.Length == 0)
            throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");

        // ── Tokens-per-second tracking (started before prefill to capture TTFT) ─
        var rateTracker = new TokenRateTracker(windowSize: 10);
        rateTracker.Start();

        // ── Prefill ───────────────────────────────────────────────────────
        int posOffset = _caches[0].Length;
        using var prefillInput = Tensor<int>.From(promptIds, 1, promptIds.Length);
        Tensor<float>? logitsTensor = _model.ForwardLastLogits(prefillInput, _caches, posOffset);

        try
        {
            int vocabSize = logitsTensor.Shape[1];
            int promptLen = promptIds.Length;

            var generatedIds = new List<int>(genCfg.MaxNewTokens);
            var decodedSoFar = new System.Text.StringBuilder();
            var rng = sampleCfg.Seed.HasValue
                ? new Random(sampleCfg.Seed.Value)
                : _defaultRng;

            for (int step = 0; step < genCfg.MaxNewTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                ReadOnlySpan<float> logitsSlice = logitsTensor.Data[..vocabSize];

                int nextId;
                if (genCfg.RepetitionPenalty != 1.0f)
                {
                    if (_penaltyScratch is null || _penaltyScratch.Length < vocabSize)
                        _penaltyScratch = new float[vocabSize];
                    Span<float> logits = _penaltyScratch.AsSpan(0, vocabSize);
                    logitsSlice.CopyTo(logits);
                    ApplyRepetitionPenalty(logits, promptIds, generatedIds,
                        genCfg.RepetitionPenalty, genCfg.RepetitionWindow);
                    nextId = Sampler.Sample(logits, sampleCfg, rng);
                }
                else
                    nextId = Sampler.Sample(logitsSlice, sampleCfg, rng);

                generatedIds.Add(nextId);

                rateTracker.RecordToken();
                TimeToFirstToken = rateTracker.TimeToFirstToken;
                TokensPerSecond = rateTracker.RollingTokensPerSecond;
                CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                if (genCfg.StopTokenIds.Contains(nextId)) break;

                _decodeTokenScratch[0] = nextId;
                string fragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true);
                decodedSoFar.Append(fragment);

                ReadOnlySpan<char> decoded = decodedSoFar.ToString().AsSpan();
                bool hitStop = false;
                foreach (string stop in genCfg.StopStrings)
                {
                    if (decoded.IndexOf(stop.AsSpan()) >= 0)
                    {
                        hitStop = true;
                        fragment = string.Empty;
                        break;
                    }
                }

                if (genCfg.Stream && fragment.Length > 0)
                    yield return fragment;

                if (hitStop) break;

                if (_caches[0].IsFull)
                {
                    int keep = genCfg is { SlidingWindowSize: > 0 }
                        ? genCfg.SlidingWindowSize
                        : _caches[0].MaxSeqLen / 2;
                    for (int i = 0; i < _caches.Length; i++)
                        _caches[i].TrimToLast(keep);
                }

                Tensor<float>? prevTensor = logitsTensor;
                logitsTensor = null;
                int newPos = posOffset + promptLen + step;

                prevTensor.Dispose();
                using var stepInput = Tensor<int>.From(_decodeTokenScratch.AsSpan(0, 1), 1, 1);
                logitsTensor = _model.ForwardLastLogits(stepInput, _caches, newPos);
            }

            if (!genCfg.Stream)
                yield return _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), skipSpecials: true);
        }
        finally
        {
            logitsTensor?.Dispose();
        }
    }

    /// <summary>
    /// Generates a full completion string without streaming.
    /// </summary>
    public async Task<string> CompleteAsync(
        string            prompt,
        SamplingConfig?   sampling   = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default)
    {
        var sb  = new System.Text.StringBuilder();
        var cfg = (generation ?? GenerationConfig.Default) with { Stream = false };

        await foreach (var fragment in GenerateAsync(prompt, sampling, cfg, cancellationToken))
            sb.Append(fragment);

        return sb.ToString();
    }

    /// <summary>Resets the KV-cache between independent generation requests.</summary>
    public void ResetCache()
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Reset();
    }

    /// <summary>KV-cache fill as a fraction of maximum capacity.</summary>
    public float CacheFillRatio => (float)_caches[0].Length / _caches[0].MaxSeqLen;

    // ── Tokens-per-second ─────────────────────────────────────────────────
    /// <summary>Rolling tokens-per-second over the last few decode steps.</summary>
    public float? TokensPerSecond { get; private set; }
    /// <summary>Cumulative tokens-per-second from the start of the current generation.</summary>
    public float? CumulativeTokensPerSecond { get; private set; }
    /// <summary>Seconds from start to first output token (includes prefill + first decode step).</summary>
    public float? TimeToFirstToken { get; private set; }

    // ── Repetition penalty ────────────────────────────────────────────────

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

/*
#if DEBUG
    private static string EscapeForDebug(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        var sb = new System.Text.StringBuilder(s.Length);
        foreach (char c in s)
        {
            if (c < 0x20 || c == 0x7f)
                sb.Append($"\\x{(int)c:X2}");
            else if (c == '\u2581')
                sb.Append('\u2581');
            else
                sb.Append(c);
        }
        return sb.ToString();
    }
#endif
*/
    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    public void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (!disposing) return;
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(StandardGenerator));
}
