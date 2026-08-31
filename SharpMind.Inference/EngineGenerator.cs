using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Inference.Chat;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <summary>
/// Token generation loop for autoregressive decoding, backed by any <see cref="IInferenceEngine"/>
/// rather than a fixed CPU <c>Transformer</c>/<c>IWorkspace</c> pair. This is
/// <see cref="StandardGenerator{T}"/>'s sampling/streaming/stop-string logic with the
/// numerics extracted behind the engine — a GPU, TPU, or any future accelerator plugs in
/// here by implementing <see cref="IInferenceEngine"/> alone; nothing in this file is
/// backend-specific. Keep this in sync with <see cref="StandardGenerator{T}"/> by hand
/// until the two are worth merging behind one shared loop.
/// </summary>
public sealed class EngineGenerator<T> : IGenerator<T> where T : IKVCacheBuilder, new()
{
    public string Name { get; init; } = "Engine";
    public IReadOnlyList<IKVCache> Caches { get; }

    private readonly IInferenceEngine _engine;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly bool _addBos, _addEos;
    private readonly Random _defaultRng;
    private readonly int[] _decodeTokenScratch = new int[1];
    private float[]? _penaltyScratch;
    private char[]? _stopCheckBuf;
    private List<int>? _generatedIds;
    private List<int>? _cacheTokens;
    private bool _disposed;

    public Action<int>? OnTokenGenerated;
    public Action<double>? PrefillProgress { get; set; }

    public float? TokensPerSecond { get; private set; }
    public float? CumulativeTokensPerSecond { get; private set; }
    public float? TimeToFirstToken { get; private set; }
    public float CacheFillRatio => _engine.MaxCacheLength == 0 ? 0f : (float)_engine.CachedLength / _engine.MaxCacheLength;
    public IReadOnlyList<int>? CurrentGeneratedIds => _generatedIds;
    public IReadOnlyList<int>? CacheTokens => _cacheTokens;

    /// <param name="numLayers">Model layer count, for the <see cref="Caches"/> view — see <see cref="EngineKVCacheView"/>.</param>
    public EngineGenerator(IInferenceEngine engine, Tokenization.Tokenizer tokenizer, bool addBos, bool addEos,
        int numLayers, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _engine = engine;
        _tokenizer = tokenizer;
        _addBos = addBos;
        _addEos = addEos;
        _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        Caches = EngineKVCacheView.ForEngine(engine, numLayers);
        // Unlike StandardGenerator, an engine handed in already warm has no way to tell us
        // what's in its cache — CacheTokens starts null (unvouched) rather than assuming empty.
        _cacheTokens = _engine.CachedLength == 0 ? [] : null;
    }

    public IAsyncEnumerable<string> GenerateAsync(string prompt, SamplingConfig? sampling = null,
        GenerationConfig? generation = null, CancellationToken cancellationToken = default)
        => GenerateFromTokensAsync(_tokenizer.Encode(prompt, _addBos, _addEos), sampling, generation, cancellationToken);

    public async IAsyncEnumerable<string> GenerateFromTokensAsync(
        int[] promptIds, SamplingConfig? sampling = null, GenerationConfig? generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        ThrowIfDisposed();

        var sampleCfg = sampling ?? SamplingConfig.Greedy;
        var genCfg = generation ?? GenerationConfig.Default;
        if (promptIds.Length == 0)
            throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");

        var rateTracker = new TokenRateTracker(windowSize: 10);
        rateTracker.Start();

        // ── Prefill ──────────────────────────────────────────────────────────
        int posOffset = _engine.CachedLength;
        if (_cacheTokens is not null && _cacheTokens.Count != posOffset) _cacheTokens = null;
        if (_cacheTokens is not null && posOffset + promptIds.Length > _engine.MaxCacheLength) _cacheTokens = null;

        ReadOnlyMemory<float> logits = _engine.Prefill(promptIds, PrefillProgress, cancellationToken);
        _cacheTokens?.AddRange(promptIds);

        int vocabSize = logits.Length;
        _generatedIds = new List<int>(genCfg.MaxNewTokens);
        var decodedSoFar = new System.Text.StringBuilder();
        var rng = sampleCfg.Seed.HasValue ? new Random(sampleCfg.Seed.Value) : _defaultRng;

        int maxStopLen = 0;
        var stopStrings = genCfg.StopStrings;
        foreach (string stop in stopStrings) if (stop.Length > maxStopLen) maxStopLen = stop.Length;
        if (maxStopLen > 0 && (_stopCheckBuf is null || _stopCheckBuf.Length < maxStopLen))
            _stopCheckBuf = new char[maxStopLen];

        var repPenalty = genCfg.RepetitionPenalty;
        var repWindow = genCfg.RepetitionWindow;
        var stopTokenIds = genCfg.StopTokenIds;
        var slidingWindowSize = genCfg.SlidingWindowSize;
        var maxNewTokens = genCfg.MaxNewTokens;
        var stream = genCfg.Stream;

        for (int step = 0; step < maxNewTokens; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (StandardGenerator.YieldBetweenTokens) await Task.Yield();

            ReadOnlySpan<float> logitsSlice = logits.Span;
            GeneratorDiagnostics.PrintTopLogits(_tokenizer, step, logitsSlice);

            int nextId;
            if (repPenalty != 1.0f)
            {
                if (_penaltyScratch is null || _penaltyScratch.Length < vocabSize) _penaltyScratch = new float[vocabSize];
                Span<float> penalized = _penaltyScratch.AsSpan(0, vocabSize);
                logitsSlice.CopyTo(penalized);                                   // copy out — the next
                ApplyRepetitionPenalty(penalized, promptIds, _generatedIds,       // DecodeStep call is free
                    repPenalty, repWindow);                                      // to reuse logits' buffer
                nextId = Sampler.Sample(penalized, sampleCfg, rng);
            }
            else
                nextId = Sampler.Sample(logitsSlice, sampleCfg, rng);            // consumed synchronously, no copy needed

            _generatedIds.Add(nextId);
            OnTokenGenerated?.Invoke(nextId);

            rateTracker.RecordToken();
            TimeToFirstToken = rateTracker.TimeToFirstToken;
            TokensPerSecond = rateTracker.RollingTokensPerSecond;
            CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

            if (stopTokenIds.Contains(nextId)) break;

            _decodeTokenScratch[0] = nextId;
            string fragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true).Replace("\uFFFD", "");
            decodedSoFar.Append(fragment);

            bool hitStop = false;
            if (maxStopLen > 0 && decodedSoFar.Length >= maxStopLen && _stopCheckBuf is not null)
            {
                int start = decodedSoFar.Length - maxStopLen;
                decodedSoFar.CopyTo(start, _stopCheckBuf, maxStopLen);
                ReadOnlySpan<char> tail = _stopCheckBuf;
                foreach (string stop in stopStrings)
                {
                    if (tail.IndexOf(stop.AsSpan()) >= 0) { hitStop = true; fragment = string.Empty; break; }
                }
            }

            if (stream && fragment.Length > 0) yield return fragment;
            if (hitStop) break;

            if (_engine.IsCacheFull)
            {
                int keep = slidingWindowSize > 0 ? slidingWindowSize : _engine.MaxCacheLength / 2;
                if (keep >= _engine.MaxCacheLength) keep = Math.Max(1, _engine.MaxCacheLength / 2);
                _engine.TrimToLast(keep);
                // Entries now sit at positions they were not computed for; nothing about the
                // cache is a prefix of any prompt any more.
                _cacheTokens = null;
            }

            // logits (the previous ReadOnlyMemory) is not read again after this point in the
            // iteration — safe for DecodeStep to reuse its backing buffer.
            logits = _engine.DecodeStep(nextId, cancellationToken);
            _cacheTokens?.Add(nextId);
        }

        if (!genCfg.Stream)
            yield return _tokenizer.Decode(CollectionsMarshal.AsSpan(_generatedIds), skipSpecials: true).Replace("\uFFFD", "");
    }

    public async Task<string> CompleteAsync(string prompt, SamplingConfig? sampling = null,
        GenerationConfig? generation = null, CancellationToken cancellationToken = default)
    {
        var sb = new System.Text.StringBuilder();
        var cfg = (generation ?? GenerationConfig.Default) with { Stream = false };
        await foreach (var fragment in GenerateAsync(prompt, sampling, cfg, cancellationToken))
            sb.Append(fragment);
        return sb.ToString();
    }

    public void ResetCache() { _engine.ResetCache(); _cacheTokens = []; }

    public void TruncateCache(int length)
    {
        _engine.TruncateCache(length);
        if (_cacheTokens is not null && _cacheTokens.Count > length)
            _cacheTokens.RemoveRange(length, _cacheTokens.Count - length);
    }

    public void SetCacheTokens(IReadOnlyList<int> tokens) => _cacheTokens = [.. tokens];

    // Reused verbatim from StandardGenerator<T> — operates on plain float spans and lists,
    // nothing CPU/backend-specific about it.
    private static void ApplyRepetitionPenalty(Span<float> logits, ReadOnlySpan<int> promptIds,
        List<int> generatedIds, float penalty, int window)
    {
        var seen = new HashSet<int>(Math.Min(promptIds.Length + generatedIds.Count, 512));
        if (window > 0)
        {
            int promptStart = Math.Max(0, promptIds.Length - window);
            RepetitionPenalty.Apply(logits, promptIds[promptStart..], penalty, seen);
            int genStart = Math.Max(0, generatedIds.Count - window);
            RepetitionPenalty.Apply(logits, CollectionsMarshal.AsSpan(generatedIds)[genStart..], penalty, seen);
            return;
        }
        RepetitionPenalty.Apply(logits, promptIds, penalty, seen);
        RepetitionPenalty.Apply(logits, CollectionsMarshal.AsSpan(generatedIds), penalty, seen);
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(EngineGenerator<>));

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _engine.Dispose();
    }
}
