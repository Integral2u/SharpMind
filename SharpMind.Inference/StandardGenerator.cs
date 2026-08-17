using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <summary>
/// Token generation loop for autoregressive decoding.
/// </summary>
public sealed class StandardGenerator<T> : IGenerator<T> where T : IKVCacheBuilder, new ()
{
    public string Name { get; init; } = "Standard";
    private readonly Transformer  _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly IKVCache[]     _caches;
    private readonly Random       _defaultRng;
    private readonly Core.Memory.Workspace _workspace;
    /// <summary>Reused decode step (<c>[1]</c>) to avoid allocating a new <see cref="int"/>[] each token.</summary>
    private readonly int[]       _decodeTokenScratch = new int[1];
    /// <summary>Cached scratch buffer for repetition-penalty copy to avoid <see cref="ArrayPool{T}.Rent"/> per token.</summary>
    private float[]?              _penaltyScratch;
    /// <summary>Diagnostic: fired for each generated token ID during the generation loop.</summary>
    public Action<int>? OnTokenGenerated;

    /// <summary>
    /// Optional progress callback for the chunked prefill phase: invoked once per
    /// chunk with the overall fraction (0..1) of the prompt prefilled so far.
    /// See <see cref="Prefill.ForwardLastLogitsChunked"/>.
    /// </summary>
    public Action<double>? PrefillProgress { get; set; }

    /// <summary>Cached scratch buffer for stop-string tail matching to avoid per-token allocation.</summary>
    private char[]?               _stopCheckBuf;
    private bool                  _disposed;
    private readonly bool _addBos;
    private List<int>? _generatedIds;
    private readonly bool _addEos;
    /// <summary>
    /// Ids resident in <see cref="_caches"/>, position by position; null once
    /// the cache's contents can no longer be vouched for (a sliding-window trim,
    /// or caches that were handed in already holding entries).
    /// </summary>
    private List<int>? _cacheTokens;

    public StandardGenerator(
        Transformer   model,
        Tokenization.Tokenizer tokenizer,
        bool addBos, bool addEos,
        IKVCache[]?   caches = null,
        int?          seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _addBos = addBos;
        _addEos = addEos;
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
                _caches[i] = new T().CreateKVCache(1, numKvHeads, maxSeqLen, headDim);
        }

        _cacheTokens = _caches[0].Length == 0 ? [] : null;
        _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        _workspace = new Core.Memory.Workspace(SharpMind.Core.Memory.Workspace.CalculateRequiredSize(model.Config.HiddenDim,model.Config.FfnDim,model.Config.VocabSize,model.Config.NumLayers, model.Config.MaxSeqLen));
    }

    // Public API

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
        int[] promptIds = _tokenizer.Encode(prompt, _addBos, _addEos);
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

        Tensor<float>? logitsTensor = null;
        try
        {
                        // Prefill (chunked so long prompts fit the workspace; see Prefill)
            int posOffset = _caches[0].Length;
            // Someone else moved the cache under us: stop vouching for it.
            if (_cacheTokens is not null && _cacheTokens.Count != posOffset) _cacheTokens = null;
            logitsTensor = Prefill.ForwardLastLogitsChunked(_model, _caches, promptIds, _workspace, PrefillProgress);
            _cacheTokens?.AddRange(promptIds);


            int vocabSize = logitsTensor.Shape[1];
            int promptLen = promptIds.Length;

            _generatedIds = new List<int>(genCfg.MaxNewTokens);
            var decodedSoFar = new System.Text.StringBuilder();
            var rng = sampleCfg.Seed.HasValue
                ? new Random(sampleCfg.Seed.Value)
                : _defaultRng;

            int maxStopLen = 0;
            var stopStrings = genCfg.StopStrings;
            foreach (string stop in stopStrings)
                if (stop.Length > maxStopLen) maxStopLen = stop.Length;
            if (maxStopLen > 0 && (_stopCheckBuf is null || _stopCheckBuf.Length < maxStopLen))
                _stopCheckBuf = new char[maxStopLen];

            var repPenalty = genCfg.RepetitionPenalty;
            var repWindow = genCfg.RepetitionWindow;
            var stopTokenIds = genCfg.StopTokenIds;
            var slidingWindowSize = genCfg.SlidingWindowSize;
            var maxNewTokens = genCfg.MaxNewTokens;
            var stream = genCfg.Stream;
            IKVCache cache0 = _caches[0];
            int cachesLen = _caches.Length;

            for (int step = 0; step < maxNewTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                ReadOnlySpan<float> logitsSlice = logitsTensor.Data[..vocabSize];
                GeneratorDiagnostics.PrintTopLogits(_tokenizer, step, logitsSlice);

                int nextId;
                if (repPenalty != 1.0f)
                {
                    if (_penaltyScratch is null || _penaltyScratch.Length < vocabSize)
                        _penaltyScratch = new float[vocabSize];
                    Span<float> logits = _penaltyScratch.AsSpan(0, vocabSize);
                    logitsSlice.CopyTo(logits);
                    ApplyRepetitionPenalty(logits, promptIds, _generatedIds,
                        repPenalty, repWindow);
                    nextId = Sampler.Sample(logits, sampleCfg, rng);
                }
                else
                    nextId = Sampler.Sample(logitsSlice, sampleCfg, rng);

                _generatedIds.Add(nextId);
                OnTokenGenerated?.Invoke(nextId);

                rateTracker.RecordToken();
                TimeToFirstToken = rateTracker.TimeToFirstToken;
                TokensPerSecond = rateTracker.RollingTokensPerSecond;
                CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                if (stopTokenIds.Contains(nextId)) break;

                _decodeTokenScratch[0] = nextId;
                string fragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true);
                decodedSoFar.Append(fragment);

                bool hitStop = false;
                if (maxStopLen > 0 && decodedSoFar.Length >= maxStopLen && _stopCheckBuf is not null)
                {
                    int start = decodedSoFar.Length - maxStopLen;
                    decodedSoFar.CopyTo(start, _stopCheckBuf, maxStopLen);
                    ReadOnlySpan<char> tail = _stopCheckBuf;
                    foreach (string stop in stopStrings)
                    {
                        if (tail.IndexOf(stop.AsSpan()) >= 0)
                        {
                            hitStop = true;
                            fragment = string.Empty;
                            break;
                        }
                    }
                }

                if (stream && fragment.Length > 0)
                {
                    // Buffer the fragment to yield outside the try-finally if needed, 
                    // but for simple Tensors, a try-finally without a catch is OK.
                    // Wait, actually yield return is allowed in try-finally, just not try-catch.
                    yield return fragment;
                }

                if (hitStop) break;

                if (cache0.IsFull)
                {
                    int keep = slidingWindowSize > 0
                        ? slidingWindowSize
                        : cache0.MaxSeqLen / 2;
                    for (int i = 0; i < cachesLen; i++)
                        _caches[i].TrimToLast(keep);
                    // Entries now sit at positions they were not computed for;
                    // nothing about the cache is a prefix of any prompt any more.
                    _cacheTokens = null;
                }

                Tensor<float>? prevTensor = logitsTensor;
                logitsTensor = null;
                int newPos = posOffset + promptLen + step;

                _workspace.Reset();
                prevTensor.Dispose();
                using var stepInput = _workspace.Rent<int>([1, 1]);
                _decodeTokenScratch[0] = nextId;
                stepInput.Data[0] = nextId;
                logitsTensor = _model.ForwardLastLogits(stepInput, _caches, newPos, _workspace);
                _cacheTokens?.Add(nextId);
            }

            if (!genCfg.Stream)
                yield return _tokenizer.Decode(CollectionsMarshal.AsSpan(_generatedIds), skipSpecials: true);
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
        _cacheTokens = [];
    }

    public IReadOnlyList<int>? CacheTokens => _cacheTokens;

    public void TruncateCache(int length)
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Truncate(length);
        if (_cacheTokens is not null && _cacheTokens.Count > length)
            _cacheTokens.RemoveRange(length, _cacheTokens.Count - length);
    }

    /// <summary>KV-cache fill as a fraction of maximum capacity.</summary>
    public float CacheFillRatio => (float)_caches[0].Length / _caches[0].MaxSeqLen;

    // Tokens-per-second
    /// <summary>Rolling tokens-per-second over the last few decode steps.</summary>
    public float? TokensPerSecond { get; private set; }
    /// <summary>Cumulative tokens-per-second from the start of the current generation.</summary>
    public float? CumulativeTokensPerSecond { get; private set; }
    /// <summary>Seconds from start to first output token (includes prefill + first decode step).</summary>
    public float? TimeToFirstToken { get; private set; }
    /// <summary>Exposes generated token IDs for diagnostics.</summary>
    public IReadOnlyList<int>? CurrentGeneratedIds => _generatedIds;

    // Repetition penalty

    private static void ApplyRepetitionPenalty(
        Span<float> logits,
        ReadOnlySpan<int> promptIds,
        List<int> _generatedIds,
        float penalty,
        int window)
    {
                // Penalize once per DISTINCT token id across the window/context (matching
        // HF/llama.cpp), NOT once per occurrence. Scaling per occurrence would raise
        // the penalty to penalty^count for common words, wiping them from the
        // distribution as generation (and context) grows.
        var seen = new HashSet<int>(Math.Min(promptIds.Length + _generatedIds.Count, 512));
        if (window > 0)
        {
            int promptStart = Math.Max(0, promptIds.Length - window);
            RepetitionPenalty.Apply(logits, promptIds[promptStart..], penalty, seen);
            int genStart = Math.Max(0, _generatedIds.Count - window);
            RepetitionPenalty.Apply(logits, CollectionsMarshal.AsSpan(_generatedIds)[genStart..], penalty, seen);
            return;
        }

        RepetitionPenalty.Apply(logits, promptIds, penalty, seen);
        RepetitionPenalty.Apply(logits, CollectionsMarshal.AsSpan(_generatedIds), penalty, seen);
    }

    // Disposal

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
        _workspace.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(StandardGenerator<>));
}
