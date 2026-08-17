using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Inference;

public sealed class SpeculativeGenerator<T> : IGenerator<T> where T : IKVCacheBuilder, new()
{
    public string Name { get; init; } = "Speculative";
    private readonly Transformer _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly IKVCache[] _caches;
    private readonly Random _defaultRng;
    private readonly Core.Memory.Workspace? _workspace;
    private readonly int[] _decodeTokenScratch = new int[1];
    private float[]? _penaltyScratch;
    private bool _disposed;

    private const int DefaultMaxDraftTokens = 4;
    private readonly bool _addBos;
    private readonly bool _addEos;

    /// <summary>
    /// Optional progress callback for the chunked prefill phase: invoked once per
    /// chunk with the overall fraction (0..1) of the prompt prefilled so far.
    /// See <see cref="Prefill.ForwardLastLogitsChunked"/>.
    /// </summary>
    public Action<double>? PrefillProgress { get; set; }

    public SpeculativeGenerator(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        bool addBos, bool addEos,
        IKVCache[]? caches = null,
        int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        _addBos = addBos;
        _addEos = addEos;
        _model = model;
        _tokenizer = tokenizer;

        if (caches != null)
        {
            _caches = caches;
        }
        else
        {
            int numLayers = model.Config.NumLayers;
            int maxSeqLen = model.Config.MaxSeqLen;
            int numKvHeads = model.Config.NumKvHeads;
            int headDim = model.Config.HeadDim;

        _caches = new IKVCache[numLayers];
        for (int i = 0; i < numLayers; i++)
            _caches[i] = new T().CreateKVCache(1, numKvHeads, maxSeqLen, headDim);
    }

    _workspace = new Core.Memory.Workspace(SharpMind.Core.Memory.Workspace.CalculateRequiredSize(model.Config.HiddenDim, model.Config.FfnDim, model.Config.VocabSize, model.Config.NumLayers, model.Config.MaxSeqLen));
    _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
}

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        int[] promptIds = _tokenizer.Encode(prompt, _addBos, _addEos);
        if (promptIds.Length == 0)
            throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");
        await foreach (var fragment in GenerateCoreAsync(promptIds, sampling, generation, DefaultMaxDraftTokens, cancellationToken))
            yield return fragment;
    }


    public async IAsyncEnumerable<string> GenerateFromTokensAsync(
        int[] promptIds,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        if (promptIds.Length == 0)
            throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");
        await foreach (var fragment in GenerateCoreAsync(promptIds, sampling, generation, DefaultMaxDraftTokens, cancellationToken))
            yield return fragment;
    }

    private async IAsyncEnumerable<string> GenerateCoreAsync(
        int[] promptIds,
        SamplingConfig? sampling,
        GenerationConfig? generation,
        int maxDraftTokens,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        ThrowIfDisposed();

        var sampleCfg = sampling ?? SamplingConfig.Greedy;
        var genCfg = generation ?? GenerationConfig.Default;

        var rateTracker = new TokenRateTracker(windowSize: 10);
        rateTracker.Start();

        int posOffset = _caches[0].Length;
        Tensor<float>? logitsTensor = Prefill.ForwardLastLogitsChunked(_model, _caches, promptIds, _workspace!, PrefillProgress);

        try
        {
            int vocabSize = logitsTensor.Shape[1];
            int promptLen = promptIds.Length;

            var generatedIds = new List<int>(genCfg.MaxNewTokens);
            var decodedSoFar = new System.Text.StringBuilder();
            var rng = sampleCfg.Seed.HasValue
                ? new Random(sampleCfg.Seed.Value)
                : _defaultRng;

            int currentPos = posOffset + promptLen;

            while (generatedIds.Count < genCfg.MaxNewTokens)
            {
                cancellationToken.ThrowIfCancellationRequested();

                ReadOnlySpan<float> logitsSlice = logitsTensor.Data[..vocabSize];
                GeneratorDiagnostics.PrintTopLogits(_tokenizer, generatedIds.Count, logitsSlice);

                if (maxDraftTokens < 1) break;

                object?[]? snapshots = null;
                if (maxDraftTokens > 1)
                {
                    snapshots = new object?[_caches.Length];
                    for (int i = 0; i < _caches.Length; i++)
                        snapshots[i] = _caches[i].Snapshot();
                }

                if (_penaltyScratch is null || _penaltyScratch.Length < vocabSize)
                    _penaltyScratch = new float[vocabSize];
                logitsTensor.Data[..vocabSize].CopyTo(_penaltyScratch.AsSpan(0, vocabSize));

                int tokensAccepted = 0;
                bool roundDone = false;

                for (int di = 0; di < maxDraftTokens && !roundDone && generatedIds.Count < genCfg.MaxNewTokens; di++)
                {
                    Span<float> curLogits = _penaltyScratch.AsSpan(0, vocabSize);

                    if (genCfg.RepetitionPenalty != 1.0f)
                        ApplyRepetitionPenalty(curLogits, promptIds, generatedIds,
                            genCfg.RepetitionPenalty, genCfg.RepetitionWindow);

                    // greedyChoice and draftToken come from the SAME logits
                    int greedyChoice = Sampler.Sample(curLogits, SamplingConfig.Greedy, rng);
                    int draftToken = Sampler.Sample(curLogits, sampleCfg, rng);

                    if (draftToken == greedyChoice)
                    {
                        // Accept draft: forward it to update cache and get next logits
                        _decodeTokenScratch[0] = draftToken;
                        Tensor<float>? prevTensor = logitsTensor;
                        logitsTensor = null;
                        _workspace?.Reset();
                        using var stepInput = _workspace != null 
                            ? _workspace.Rent<int>([1, 1]) 
                            : Tensor<int>.From(_decodeTokenScratch.AsSpan(0, 1), 1, 1);
                        if (_workspace != null) stepInput.Data[0] = draftToken;
                        logitsTensor = _model.ForwardLastLogits(stepInput, _caches, currentPos, _workspace);
                        currentPos++;
                        prevTensor?.Dispose();

                        generatedIds.Add(draftToken);
                        rateTracker.RecordToken();
                        TimeToFirstToken = rateTracker.TimeToFirstToken;
                        TokensPerSecond = rateTracker.RollingTokensPerSecond;
                        CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;
                        tokensAccepted++;

                        string fragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true);
                        decodedSoFar.Append(fragment);

                        bool hitStop = false;
                        foreach (string stop in genCfg.StopStrings)
                        {
                            if (StringBuilderContains(decodedSoFar, stop))
                            { hitStop = true; fragment = string.Empty; break; }
                        }

                        if (genCfg.Stream && fragment.Length > 0)
                            yield return fragment;

                        if (hitStop || genCfg.StopTokenIds.Contains(draftToken))
                        { roundDone = true; break; }

                        logitsTensor.Data[..vocabSize].CopyTo(_penaltyScratch.AsSpan(0, vocabSize));

                        if (_caches[0].IsFull)
                        {
                            int keep = genCfg is { SlidingWindowSize: > 0 }
                                ? genCfg.SlidingWindowSize
                                : _caches[0].MaxSeqLen / 2;
                            for (int i = 0; i < _caches.Length; i++)
                                _caches[i].TrimToLast(keep);
                        }
                    }
                    else
                    {
                        // Reject: restore cache and emit the model's greedy choice
                        if (snapshots != null)
                        {
                            for (int i = 0; i < _caches.Length; i++)
                                _caches[i].Restore(snapshots[i]);
                        }
                        currentPos = posOffset + promptLen + generatedIds.Count;

                        int correctionToken = greedyChoice;
                        generatedIds.Add(correctionToken);
                        rateTracker.RecordToken();
                        TimeToFirstToken = rateTracker.TimeToFirstToken;
                        TokensPerSecond = rateTracker.RollingTokensPerSecond;
                        CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                        _decodeTokenScratch[0] = correctionToken;
                        string corrFragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true);
                        decodedSoFar.Append(corrFragment);

                        if (genCfg.Stream && corrFragment.Length > 0)
                            yield return corrFragment;

                        // Forward correction token to get next logits
                        Tensor<float>? prevTensor = logitsTensor;
                        logitsTensor = null;
                        _workspace?.Reset();
                        using var corrInput = _workspace != null 
                            ? _workspace.Rent<int>([1, 1]) 
                            : Tensor<int>.From(_decodeTokenScratch.AsSpan(0, 1), 1, 1);
                        if (_workspace != null) corrInput.Data[0] = correctionToken;
                        logitsTensor = _model.ForwardLastLogits(corrInput, _caches, currentPos, _workspace);
                        currentPos++;
                        prevTensor?.Dispose();
                        logitsTensor.Data[..vocabSize].CopyTo(_penaltyScratch.AsSpan(0, vocabSize));

                        roundDone = true;

                        if (genCfg.StopTokenIds.Contains(correctionToken))
                            break;
                    }
                }

                // If the inner loop stopped on a stop token, end generation entirely.
                if (generatedIds.Count > 0 && genCfg.StopTokenIds.Contains(generatedIds[^1]))
                    break;

                if (!roundDone && tokensAccepted == maxDraftTokens)
                {
                    // All drafts accepted: emit the bonus token (greedy sample from last logits)
                    int bonusToken = Sampler.Sample(_penaltyScratch.AsSpan(0, vocabSize), SamplingConfig.Greedy, rng);
                    generatedIds.Add(bonusToken);
                    rateTracker.RecordToken();
                    TimeToFirstToken = rateTracker.TimeToFirstToken;
                    TokensPerSecond = rateTracker.RollingTokensPerSecond;
                    CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                    _decodeTokenScratch[0] = bonusToken;
                    string bonusFragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true);
                    decodedSoFar.Append(bonusFragment);

                    if (genCfg.Stream && bonusFragment.Length > 0)
                        yield return bonusFragment;

                    if (genCfg.StopTokenIds.Contains(bonusToken))
                        break;

                    Tensor<float>? bonusPrev = logitsTensor;
                    logitsTensor = null;
                    _workspace?.Reset();
                    using var bonusInput = _workspace != null 
                        ? _workspace.Rent<int>([1, 1]) 
                        : Tensor<int>.From(_decodeTokenScratch.AsSpan(0, 1), 1, 1);
                    if (_workspace != null) bonusInput.Data[0] = bonusToken;
                    logitsTensor = _model.ForwardLastLogits(bonusInput, _caches, currentPos, _workspace);
                    currentPos++;
                    bonusPrev?.Dispose();
                }
            }

            if (!genCfg.Stream && decodedSoFar.Length > 0)
                yield return _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), skipSpecials: true);
        }
        finally
        {
            logitsTensor?.Dispose();
        }
    }

    private static bool StringBuilderContains(System.Text.StringBuilder sb, ReadOnlySpan<char> value)
    {
        if (value.IsEmpty) return true;
        if (sb.Length < value.Length) return false;
        for (int i = 0; i <= sb.Length - value.Length; i++)
        {
            bool match = true;
            for (int j = 0; j < value.Length; j++)
            {
                if (sb[i + j] != value[j]) { match = false; break; }
            }
            if (match) return true;
        }
        return false;
    }

    private static void ApplyRepetitionPenalty(
        Span<float> logits,
        int[] promptIds,
        List<int> generatedIds,
        float penalty,
        int window)
    {
        if (penalty == 1.0f) return;
        // Once per DISTINCT id across prompt + generated (windowed), matching the
        // StandardGenerator / HF reference. The previous code only scaled negative
        // logits per occurrence, so repeated likely tokens were never suppressed.
        var seen = new HashSet<int>(Math.Min(promptIds.Length + generatedIds.Count, 512));
        int genStart = Math.Max(0, generatedIds.Count - (window > 0 ? window : generatedIds.Count));
        RepetitionPenalty.Apply(logits, promptIds, penalty, seen);
        RepetitionPenalty.Apply(logits, CollectionsMarshal.AsSpan(generatedIds)[genStart..], penalty, seen);
    }

    public void ResetCache()
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Reset();
    }

    public void TruncateCache(int length)
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Truncate(length);
    }

    public float CacheFillRatio => (float)_caches[0].Length / _caches[0].MaxSeqLen;

    public float? TokensPerSecond { get; private set; }
    public float? CumulativeTokensPerSecond { get; private set; }
    public float? TimeToFirstToken { get; private set; }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _workspace?.Dispose();
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(SpeculativeGenerator<>));
}
