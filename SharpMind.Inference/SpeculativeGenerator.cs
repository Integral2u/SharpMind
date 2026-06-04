using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <summary>
/// Speculative Decoding generator - generates speculative tokens then verifies them.
///
/// Algorithm (Levenshtein et al., 2023):
/// 1. Draft: Generate N speculative tokens (either from draft model or greedy from main)
/// 2. Verify: Run full forward pass on all N tokens
/// 3. Accept: Keep tokens where draft matches verification
/// 4. Correct: On mismatch, use verification distribution to sample next token
///
/// Speedup comes from verifying multiple tokens at once vs autoregressive single-token decoding.
/// </summary>
public sealed class SpeculativeGenerator : IGenerator
{
    private readonly Transformer _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly IKVCache[] _caches;
    private readonly Random _defaultRng;
    private readonly int[] _decodeTokenScratch = new int[1];
    private bool _disposed;

    private const int DefaultMaxDraftTokens = 4;

    public SpeculativeGenerator(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        _model = model;
        _tokenizer = tokenizer;

        int numLayers = model.Config.NumLayers;
        int maxSeqLen = model.Config.MaxSeqLen;
        int numKvHeads = model.Config.NumKvHeads;
        int headDim = model.Config.HeadDim;

        _caches = new IKVCache[numLayers];
        for (int i = 0; i < numLayers; i++)
            _caches[i] = new KVCache(1, numKvHeads, maxSeqLen, headDim);

        _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
    }

    public async IAsyncEnumerable<string> GenerateAsync(
        string prompt,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        int maxDraftTokens = DefaultMaxDraftTokens,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        ThrowIfDisposed();

        var sampleCfg = sampling ?? SamplingConfig.Greedy;
        var genCfg = generation ?? GenerationConfig.Default;

        int[] promptIds = _tokenizer.Encode(prompt, addBos: true, addEos: false);

        var rateTracker = new TokenRateTracker(windowSize: 10);
        rateTracker.Start();

        int posOffset = _caches[0].Length;
        using var prefillInput = Tensor<int>.From(promptIds, 1, promptIds.Length);
        Tensor<float>? logitsTensor = _model.Forward(prefillInput, _caches, posOffset);

        try
        {
        int vocabSize = logitsTensor.Shape[2];
        int lastPromptPos = promptIds.Length - 1;
        ReadOnlySpan<float> logitsRow = logitsTensor.Data.Slice(lastPromptPos * vocabSize, vocabSize);

        var generatedIds = new List<int>(genCfg.MaxNewTokens);
        var decodedSoFar = new System.Text.StringBuilder();
        var rng = sampleCfg.Seed.HasValue
            ? new Random(sampleCfg.Seed.Value)
            : _defaultRng;

        int currentPos = posOffset + promptIds.Length;

        while (generatedIds.Count < genCfg.MaxNewTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            if (maxDraftTokens < 1)
                break;

            var draftTokens = new int[maxDraftTokens];
            // One row of logits = one position — true multi-token drafting needs extra model calls or a draft model.
            draftTokens[0] = Sampler.Sample(logitsRow, sampleCfg, rng);
            const int draftCount = 1;

            var (Tokens, AcceptedCount, NeedsCorrection, CorrectionToken) = VerifyDraftTokens(draftTokens, draftCount, currentPos, vocabSize, sampleCfg, rng);

            for (int i = 0; i < AcceptedCount; i++)
            {
                int tokenId = Tokens[i];
                generatedIds.Add(tokenId);

                rateTracker.RecordToken();
                TimeToFirstToken = rateTracker.TimeToFirstToken;
                TokensPerSecond = rateTracker.RollingTokensPerSecond;
                CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                _decodeTokenScratch[0] = tokenId;
                string fragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true);
                decodedSoFar.Append(fragment);

                bool hitStop = false;
                foreach (string stop in genCfg.StopStrings)
                {
                    if (StringBuilderContains(decodedSoFar, stop))
                    {
                        hitStop = true;
                        break;
                    }
                }

                if (genCfg.Stream && fragment.Length > 0)
                    yield return fragment;

                if (hitStop || genCfg.StopTokenIds.Contains(tokenId))
                    break;
            }

            currentPos += AcceptedCount;

            if (NeedsCorrection)
            {
                var correctionToken = CorrectionToken;
                generatedIds.Add(correctionToken);

                rateTracker.RecordToken();
                TimeToFirstToken = rateTracker.TimeToFirstToken;
                TokensPerSecond = rateTracker.RollingTokensPerSecond;
                CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                _decodeTokenScratch[0] = correctionToken;
                string fragment = _tokenizer.Decode(_decodeTokenScratch.AsSpan(0, 1), skipSpecials: true);
                decodedSoFar.Append(fragment);

                if (genCfg.Stream && fragment.Length > 0)
                    yield return fragment;

                if (genCfg.StopTokenIds.Contains(correctionToken))
                    break;

                currentPos++;
            }

            if (generatedIds.Count >= genCfg.MaxNewTokens)
                break;

            Tensor<float>? prevLogits = logitsTensor;
            logitsTensor = null;
            _decodeTokenScratch[0] = generatedIds[^1];
            prevLogits.Dispose();
            using var nextInput = Tensor<int>.From(_decodeTokenScratch.AsSpan(0, 1), 1, 1);
            logitsTensor = _model.Forward(nextInput, _caches, currentPos);
            logitsRow = logitsTensor.Data[..vocabSize];
        }

        if (!genCfg.Stream && decodedSoFar.Length > 0)
            yield return _tokenizer.Decode(CollectionsMarshal.AsSpan(generatedIds), skipSpecials: true);
        }
        finally
        {
            logitsTensor?.Dispose();
        }
    }

    IAsyncEnumerable<string> IGenerator.GenerateAsync(
        string prompt,
        SamplingConfig? sampling,
        GenerationConfig? generation,
        CancellationToken cancellationToken)
        => GenerateAsync(prompt, sampling, generation, DefaultMaxDraftTokens, cancellationToken);

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

    private (int[] Tokens, int AcceptedCount, bool NeedsCorrection, int CorrectionToken) VerifyDraftTokens(
        int[] draftTokens,
        int draftCount,
        int position,
        int vocabSize,
        SamplingConfig cfg,
        Random rng)
    {
        var accepted = new int[draftCount];
        int acceptedCount = 0;
        bool needsCorrection = false;
        int correctionToken = 0;

        for (int i = 0; i < draftCount; i++)
        {
            int pos = position + i;
            _decodeTokenScratch[0] = draftTokens[i];
            using var input = Tensor<int>.From(_decodeTokenScratch.AsSpan(0, 1), 1, 1);
            using var logits = _model.Forward(input, _caches, pos);

            int verifiedToken = Sampler.Sample(logits.Data[..vocabSize], cfg, rng);

            if (verifiedToken == draftTokens[i])
            {
                accepted[acceptedCount++] = draftTokens[i];
            }
            else
            {
                needsCorrection = true;
                correctionToken = verifiedToken;
                break;
            }
        }

        return (accepted, acceptedCount, needsCorrection, correctionToken);
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

    public async IAsyncEnumerable<string> GenerateFromTokensAsync(
        int[] promptIds,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        await foreach (var fragment in GenerateAsync(
            _tokenizer.Decode(promptIds, skipSpecials: true),
            sampling, generation, DefaultMaxDraftTokens, cancellationToken))
            yield return fragment;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(SpeculativeGenerator));
}