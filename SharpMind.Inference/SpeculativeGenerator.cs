using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;

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
public sealed class SpeculativeGenerator : IDisposable
{
    private readonly Transformer _model;
    private readonly SharpMind.Tokenization.Tokenizer _tokenizer;
    private readonly InferenceOps _ops;
    private readonly KVCache[] _caches;
    private readonly Random _defaultRng;
    private bool _disposed;

    private const int DefaultMaxDraftTokens = 4;

    public SpeculativeGenerator(
        Transformer model,
        SharpMind.Tokenization.Tokenizer tokenizer,
        InferenceOps ops,
        int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(ops);

        _model = model;
        _tokenizer = tokenizer;
        _ops = ops;

        int numLayers = model.Config.NumLayers;
        int maxSeqLen = model.Config.MaxSeqLen;
        int numKvHeads = model.Config.NumKvHeads;
        int headDim = model.Config.HeadDim;

        _caches = new KVCache[numLayers];
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

        int posOffset = _caches[0].Length;
        using var prefillInput = Tensor<int>.From(promptIds, 1, promptIds.Length);
        using var prefillLogits = _model.Forward(prefillInput, _caches, posOffset);

        int vocabSize = prefillLogits.Shape[2];
        int lastPromptPos = promptIds.Length - 1;
        var logitsSlice = prefillLogits.Data.Slice(lastPromptPos * vocabSize, vocabSize);

        var generatedIds = new List<int>(genCfg.MaxNewTokens);
        var decodedSoFar = string.Empty;
        var rng = sampleCfg.Seed.HasValue
            ? new Random(sampleCfg.Seed.Value)
            : _defaultRng;

        int currentPos = posOffset + promptIds.Length;

        while (generatedIds.Count < genCfg.MaxNewTokens)
        {
            cancellationToken.ThrowIfCancellationRequested();

            var draftTokens = new int[maxDraftTokens];
            int draftCount = 0;

            for (int d = 0; d < maxDraftTokens && generatedIds.Count + d < genCfg.MaxNewTokens; d++)
            {
                var slice = logitsSlice.Slice(d * vocabSize, vocabSize).ToArray();
                draftTokens[d] = Sampler.Sample(slice, sampleCfg, rng);
                
                if (genCfg.StopTokenIds.Contains(draftTokens[d]))
                {
                    draftTokens[d] = draftTokens[d];
                    draftCount = d + 1;
                    break;
                }
                draftCount = d + 1;
            }

            if (draftCount == 0) break;

            var verifiedTokens = VerifyDraftTokens(draftTokens, draftCount, currentPos, vocabSize, sampleCfg, rng);

            for (int i = 0; i < verifiedTokens.AcceptedCount; i++)
            {
                int tokenId = verifiedTokens.Tokens[i];
                generatedIds.Add(tokenId);

                string fragment = _tokenizer.Decode([tokenId], skipSpecials: true);
                decodedSoFar += fragment;

                bool hitStop = false;
                foreach (string stop in genCfg.StopStrings)
                {
                    if (decodedSoFar.Contains(stop, StringComparison.Ordinal))
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

            currentPos += verifiedTokens.AcceptedCount;

            if (verifiedTokens.NeedsCorrection)
            {
                var correctionToken = verifiedTokens.CorrectionToken;
                generatedIds.Add(correctionToken);

                string fragment = _tokenizer.Decode([correctionToken], skipSpecials: true);
                decodedSoFar += fragment;

                if (genCfg.Stream && fragment.Length > 0)
                    yield return fragment;

                if (genCfg.StopTokenIds.Contains(correctionToken))
                    break;

                currentPos++;
            }

            if (generatedIds.Count >= genCfg.MaxNewTokens)
                break;

            using var nextInput = Tensor<int>.From([generatedIds[^1]], 1, 1);
            using var nextLogits = _model.Forward(nextInput, _caches, currentPos);
            logitsSlice = nextLogits.Data[..vocabSize];
        }

        if (!genCfg.Stream && decodedSoFar.Length > 0)
            yield return _tokenizer.Decode([.. generatedIds], skipSpecials: true);
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
            using var input = Tensor<int>.From([draftTokens[i]], 1, 1);
            using var logits = _model.Forward(input, _caches, pos);

            var tokenLogits = logits.Data[..vocabSize].ToArray();
            int verifiedToken = Sampler.Sample(tokenLogits, cfg, rng);

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

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(SpeculativeGenerator));
}