using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Layers;

namespace SharpMind.Inference;

public sealed class MedusaGenerator<T> : IGenerator<T> where T : IKVCacheBuilder, new()
{
    private readonly Transformer _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly IKVCache[] _caches;
    private readonly Random _defaultRng;
    private readonly SharpMind.Core.Memory.Workspace _workspace;
    private readonly MedusaHeads _medusaHeads;
    private readonly float[] _normedHiddenScratch;
    private readonly int[] _draftScratch;
    private bool _disposed;
    private readonly bool _addBos;
    private readonly bool _addEos;

    private const int DefaultNumHeads = 3;

    public MedusaGenerator(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        bool addBos, bool addEos,
        MedusaHeads medusaHeads,
        IKVCache[]? caches = null,
        int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(medusaHeads);
        _addBos = addBos;
        _addEos = addEos;
        _model = model;
        _tokenizer = tokenizer;
        _medusaHeads = medusaHeads;

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

        _workspace = new SharpMind.Core.Memory.Workspace(
            SharpMind.Core.Memory.Workspace.CalculateRequiredSize(
                model.Config.HiddenDim, model.Config.FfnDim, model.Config.VocabSize,
                model.Config.NumLayers, model.Config.MaxSeqLen));

        _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        _normedHiddenScratch = new float[model.Config.HiddenDim];
        _draftScratch = new int[DefaultNumHeads + 1];
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
        await foreach (var fragment in GenerateCoreAsync(promptIds, sampling, generation, cancellationToken))
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
        await foreach (var fragment in GenerateCoreAsync(promptIds, sampling, generation, cancellationToken))
            yield return fragment;
    }

    private async IAsyncEnumerable<string> GenerateCoreAsync(
        int[] promptIds,
        SamplingConfig? sampling,
        GenerationConfig? generation,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        ThrowIfDisposed();

        var sampleCfg = sampling ?? SamplingConfig.Greedy;
        var genCfg = generation ?? GenerationConfig.Default;

        var rateTracker = new TokenRateTracker(windowSize: 10);
        rateTracker.Start();

        int hiddenDim = _model.Config.HiddenDim;
        int vocabSize = _model.Config.VocabSize;
        int numHeads = _medusaHeads.NumHeads;
        int draftLen = numHeads + 1;

        // ── Prefill ──
        _workspace.Reset();
        int posOffset = _caches[0].Length;
        using var prefillInput = _workspace.Rent<int>([1, promptIds.Length]);
        promptIds.CopyTo(prefillInput.Data);
        Tensor<float>? logitsTensor = _model.ForwardLastLogits(prefillInput, _caches, posOffset, _workspace);

        try
        {
            int promptLen = promptIds.Length;
            var generatedIds = new List<int>(genCfg.MaxNewTokens);
            var decodedSoFar = new System.Text.StringBuilder();
            var rng = sampleCfg.Seed.HasValue
                ? new Random(sampleCfg.Seed.Value)
                : _defaultRng;

            int currentPos = posOffset + promptLen;
            int[] scratchOne = new int[1];

            // After prefill, extract normed hidden for the last prompt position
            RefillNormedHidden(promptLen - 1);

            while (generatedIds.Count < genCfg.MaxNewTokens)
            {
                cancellationToken.ThrowIfCancellationRequested();

                ReadOnlySpan<float> curLogits = logitsTensor.Data[..vocabSize];

                // 1. Greedy sample from current logits
                int token0 = Sampler.Sample(curLogits, SamplingConfig.Greedy, rng);
                _draftScratch[0] = token0;

                // 2. Medusa heads predict tokens 1..K
                if (numHeads > 0)
                {
                    _medusaHeads.Predict(_normedHiddenScratch, _draftScratch.AsSpan(1, numHeads), _model.Ops);
                }

                // 3. Save cache length before verification
                int cacheLenBefore = _caches[0].Length;

                // 4. Forward draft sequence
                Tensor<float> prevLogits = logitsTensor;
                logitsTensor = null;
                _workspace.Reset();
                using var draftInput = _workspace.Rent<int>([1, draftLen]);
                for (int i = 0; i < draftLen; i++)
                    draftInput.Data[i] = _draftScratch[i];
                var verifLogits = _model.Forward(draftInput, _caches, currentPos, _workspace);
                prevLogits.Dispose();
                // verifLogits: [1, draftLen, vocabSize]
                // _cachedHidden: [1, draftLen, hiddenDim] (pre-norm arch output)

                // 5. Verify and accept prefix
                int accepted = 1;
                for (int i = 0; i < numHeads; i++)
                {
                    int verifGreedy = Sampler.Sample(
                        verifLogits.Data.Slice(i * vocabSize, vocabSize),
                        SamplingConfig.Greedy, rng);
                    if (verifGreedy == _draftScratch[i + 1])
                        accepted++;
                    else
                        break;
                }

                // 6. Emit accepted tokens (one by one for streaming)
                bool stopHit = false;
                for (int i = 0; i < accepted; i++)
                {
                    int tid = _draftScratch[i];
                    generatedIds.Add(tid);
                    rateTracker.RecordToken();
                    TimeToFirstToken = rateTracker.TimeToFirstToken;
                    TokensPerSecond = rateTracker.RollingTokensPerSecond;
                    CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                    scratchOne[0] = tid;
                    string fragment = _tokenizer.Decode(scratchOne.AsSpan(0, 1), skipSpecials: true);
                    decodedSoFar.Append(fragment);

                    ReadOnlySpan<char> decoded = decodedSoFar.ToString().AsSpan();
                    foreach (string stop in genCfg.StopStrings)
                    {
                        if (decoded.IndexOf(stop.AsSpan()) >= 0)
                        {
                            stopHit = true;
                            break;
                        }
                    }

                    if (genCfg.Stream && fragment.Length > 0 && !stopHit)
                        yield return fragment;

                    if (stopHit || genCfg.StopTokenIds.Contains(tid))
                        break;
                }
                if (stopHit) break;
                if (generatedIds.Count > 0 && genCfg.StopTokenIds.Contains(generatedIds[^1]))
                    break;

                // 7. Prepare next round's starting state
                if (accepted == draftLen)
                {
                    // All draft tokens accepted — bonus token
                    int bonus = Sampler.Sample(
                        verifLogits.Data.Slice(numHeads * vocabSize, vocabSize),
                        SamplingConfig.Greedy, rng);
                    generatedIds.Add(bonus);
                    rateTracker.RecordToken();
                    TimeToFirstToken = rateTracker.TimeToFirstToken;
                    TokensPerSecond = rateTracker.RollingTokensPerSecond;
                    CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                    scratchOne[0] = bonus;
                    string bonusFragment = _tokenizer.Decode(scratchOne.AsSpan(0, 1), skipSpecials: true);
                    decodedSoFar.Append(bonusFragment);

                    if (genCfg.Stream && bonusFragment.Length > 0)
                        yield return bonusFragment;

                    if (genCfg.StopTokenIds.Contains(bonus))
                        break;

                    if (_caches[0].IsFull)
                    {
                        int keep = genCfg is { SlidingWindowSize: > 0 }
                            ? genCfg.SlidingWindowSize
                            : _caches[0].MaxSeqLen / 2;
                        for (int i = 0; i < _caches.Length; i++)
                            _caches[i].TrimToLast(keep);
                    }

                    currentPos += draftLen;
                    // Forward bonus token to get clean state for next round
                    Tensor<float>? bonusPrev = verifLogits;
                    verifLogits = null;
                    _workspace.Reset();
                    using var bonusInput = _workspace.Rent<int>([1, 1]);
                    bonusInput.Data[0] = bonus;
                    logitsTensor = _model.ForwardLastLogits(bonusInput, _caches, currentPos, _workspace);
                    bonusPrev.Dispose();
                    currentPos++;

                    // After single-token ForwardLastLogits, _cachedHidden is [1, 1, H] and normed
                    var ch = _model.LastCachedHidden;
                    if (ch != null)
                        ch.Data[..hiddenDim].CopyTo(_normedHiddenScratch);
                }
                else
                {
                    // Partial acceptance: trim cache and extract state from verification
                    int lastAcceptedIdx = accepted - 1;

                    for (int i = 0; i < _caches.Length; i++)
                        _caches[i].TrimToLast(cacheLenBefore + accepted);

                    if (_caches[0].IsFull)
                    {
                        int keep = genCfg is { SlidingWindowSize: > 0 }
                            ? genCfg.SlidingWindowSize
                            : _caches[0].MaxSeqLen / 2;
                        for (int i = 0; i < _caches.Length; i++)
                            _caches[i].TrimToLast(keep);
                    }

                    // Next round's logits = verif_logits[lastAcceptedIdx]
                    var nextLogits = new Tensor<float>(1, vocabSize);
                    verifLogits.Data.Slice(lastAcceptedIdx * vocabSize, vocabSize)
                        .CopyTo(nextLogits.Data);
                    logitsTensor = nextLogits;
                    verifLogits.Dispose();

                    // Refill normed hidden for next Medusa prediction
                    RefillNormedHidden(lastAcceptedIdx);

                    currentPos += accepted;
                }
            }

            if (!genCfg.Stream)
            {
                string full = _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), skipSpecials: true);
                if (full.Length > 0)
                    yield return full;
            }
        }
        finally
        {
            logitsTensor?.Dispose();
        }
    }

    /// <summary>
    /// Copies the row at <paramref name="rowIndex"/> from the model's cached hidden
    /// state and applies final norm in-place on <see cref="_normedHiddenScratch"/>.
    /// </summary>
    private void RefillNormedHidden(int rowIndex)
    {
        var cachedHidden = _model.LastCachedHidden;
        if (cachedHidden == null) return;
        int hiddenDim = _model.Config.HiddenDim;
        cachedHidden.Data.Slice(rowIndex * hiddenDim, hiddenDim).CopyTo(_normedHiddenScratch);

        using var normTemp = new Tensor<float>(1, hiddenDim);
        _normedHiddenScratch.CopyTo(normTemp.Data);
        _model.FinalNorm.ForwardInPlace(normTemp);
        normTemp.Data.CopyTo(_normedHiddenScratch);
    }

    public void ResetCache()
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Reset();
    }

    public float CacheFillRatio => (float)_caches[0].Length / _caches[0].MaxSeqLen;

    public float? TokensPerSecond { get; private set; }
    public float? CumulativeTokensPerSecond { get; private set; }
    public float? TimeToFirstToken { get; private set; }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _medusaHeads.Dispose();
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
        _workspace.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(MedusaGenerator<T>));
}
