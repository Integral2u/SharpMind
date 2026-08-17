// Medusa-style speculative decoding generator.
//
// Core idea:
//   Ordinary autoregressive decoding processes one token per forward pass.
//   Medusa-style speculative decoding trains K small "draft heads" that each
//   predict a token at a different future offset from a *single* hidden state.
//   All K+1 draft tokens (the LM head's own greedy choice + K head predictions)
//   are then verified in ONE forward pass.  If the model agrees with the heads,
//   multiple tokens are accepted per forward pass instead of one.
//
// Verification pattern:
//   Round start:  logits predict token at position P  (cache length = P)
//     1.  greedy = argmax(logits)  →  token_0 at position P
//     2.  Medusa heads from normed hidden state at position P-1
//         →  token_1 (head 0), token_2 (head 1), ..., token_K (head K-1)
//     3.  Draft = [token_0, token_1, ..., token_K]  (length K+1)
//     4.  Forward(draft, startPos=P) processes all K+1 tokens as a batch:
//           verif_logits[i] predicts what comes AFTER draft[i]
//           (i.e. the token at position P+i+1)
//     5.  Verify:
//           accepted = 1  (token_0 is the LM head's own greedy choice, always accepted)
//           for i = 0..K-1:
//             if argmax(verif_logits[i]) == draft[i+1] → accepted++
//             else break
//     6.  Emit draft[0 .. accepted-1]  (positions P .. P+accepted-1)
//     7.  Prepare next round:
//           IF accepted == K+1 (all accepted):
//             bonus = argmax(verif_logits[K])   ← "free" extra token
//             emit bonus
//             ForwardLastLogits(bonus)          ← clean state for next round
//           ELSE:
//             Trim KV cache to P + accepted     ← discard rejected suffixes
//             Next round's logits = verif_logits[accepted-1]
//             Next round's hidden state = norm(_cachedHidden row accepted-1)
//
// Why this helps:
//   Without Medusa:  M tokens → M forward passes (one per token).
//   With Medusa:     If M ≤ K+1 accepted, only 1 batch forward is needed.
//                    If M = K+1, we also get a bonus token "for free",
//                    totalling K+2 tokens for 2 forward passes.
//
//   In the ideal case (well-trained heads, acceptance near K+1):
//     Speedup ≈ (K+2) / 2   (e.g. 5/2 = 2.5× for K=3)
//
//   When acceptance is low, Medusa still produces correct output because
//   every rejected token is caught by validation — correctness is identical
//   to greedy decoding, with only throughput affected.
//
// Current limitations / known issues:
//   - Heads are randomly initialised — calibration must be run separately
//     (MedusaHeads.Calibrate) before the generator produces speedups.
//   - The "bonus" all-accepted path does one extra ForwardLastLogits to
//     reset the hidden state; this slightly dilutes the peak speedup.
//   - Sliding window (TrimToLast) is used when the KV cache fills, but
//     Medusa interacts poorly with very short windows — the draft tokens
//     may reference positions that were trimmed out of attention range.

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Layers;

namespace SharpMind.Inference;

public sealed class MedusaGenerator<T> : IGenerator<T> where T : IKVCacheBuilder, new()
{
    public string Name { get; init; } = "Medusa";
    private readonly Transformer _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly IKVCache[] _caches;
    private readonly Random _defaultRng;
    private readonly Core.Memory.Workspace _workspace;
    private readonly MedusaHeads _medusaHeads;

    // _normedHiddenScratch caches the RMS-normed hidden state that feeds the Medusa heads.
    // It is refilled every round from _cachedHidden (maintained by Transformer) after
    // verifying and accepting a prefix.  We keep a float[] copy rather than a Tensor
    // because the Medusa heads' Predict() takes a Span<float> and we need the normed
    // values to survive workspace resets.
    private readonly float[] _normedHiddenScratch;

    // Scratch buffer for the K+1 draft tokens.  Index 0 = LM head greedy choice,
    // indices 1..K = Medusa head predictions.  The Predict() call writes directly
    // into slots 1..K.
    private readonly int[] _draftScratch;
    private bool _disposed;
    private readonly bool _addBos;
    private readonly bool _addEos;

    private const int DefaultNumHeads = 3;

    /// <summary>
    /// Optional progress callback for the chunked prefill phase: invoked once per
    /// chunk with the overall fraction (0..1) of the prompt prefilled so far.
    /// See <see cref="Prefill.ForwardLastLogitsChunked"/>.
    /// </summary>
    public Action<double>? PrefillProgress { get; set; }

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

        _workspace = new Core.Memory.Workspace(
            Core.Memory.Workspace.CalculateRequiredSize(
                model.Config.HiddenDim, model.Config.FfnDim, model.Config.VocabSize,
                model.Config.NumLayers, model.Config.MaxSeqLen));

        _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        _normedHiddenScratch = new float[model.Config.HiddenDim];
        _draftScratch = new int[DefaultNumHeads + 1];
    }

    // IGenerator<T> implementation

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

    // Core generation loop

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
        int draftLen = numHeads + 1; // LM-head greedy + K head predictions

        // Prefill
        // Process the full prompt (in chunks that fit the workspace; see
        // Prefill). After this, the KV cache has promptLen entries and
        // logitsTensor predicts the very next token.
        int posOffset = _caches[0].Length;
        Tensor<float>? logitsTensor = Prefill.ForwardLastLogitsChunked(
            _model, _caches, promptIds, _workspace, PrefillProgress);

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

            // Extract the normed hidden state for the last prompt position.
            // After a chunked prefill _cachedHidden holds only the last prefill
            // chunk [1, chunkLen, H] pre-norm, so the last prompt token is its
            // final row; RefillNormedHidden copies that row, norms it, and stores
            // it into _normedHiddenScratch for the first Medusa prediction.
            RefillNormedHidden((_model.LastCachedHidden?.Shape[1] ?? 1) - 1);

            // Decode loop
            while (generatedIds.Count < genCfg.MaxNewTokens)
            {
                cancellationToken.ThrowIfCancellationRequested();

                ReadOnlySpan<float> curLogits = logitsTensor.Data[..vocabSize];
                GeneratorDiagnostics.PrintTopLogits(_tokenizer, generatedIds.Count, curLogits);

                // 1. Greedy sample: the LM head's best guess for the next token.
                //    This is always the first draft token.
                int token0 = Sampler.Sample(curLogits, SamplingConfig.Greedy, rng);
                _draftScratch[0] = token0;

                // 2. Run the Medusa draft heads.  Each head is a small MLP
                //    (Linear + SiLU + shared LM-head projection) that predicts
                //    a token at a different future offset from the *same* normed
                //    hidden state.  Predictions go into _draftScratch[1..K].
                if (numHeads > 0)
                {
                    _medusaHeads.Predict(
                        _normedHiddenScratch,
                        _draftScratch.AsSpan(1, numHeads));
                }

                // 3. Snapshot cache length before verification so we can trim
                //    back if some draft tokens are rejected.
                int cacheLenBefore = _caches[0].Length;

                // 4. Batch-forward all K+1 draft tokens.
                //    Forward(draft) extends the KV cache by K+1 positions and
                //    returns logits [1, K+1, V].  verif_logits[i] predicts what
                //    token comes AFTER draft[i], i.e. the token at position
                //    (currentPos + i + 1).
                Tensor<float> prevLogits = logitsTensor;
                logitsTensor = null;
                _workspace.Reset();
                using var draftInput = _workspace.Rent<int>([1, draftLen]);
                for (int i = 0; i < draftLen; i++)
                    draftInput.Data[i] = _draftScratch[i];
                var verifLogits = _model.Forward(
                    draftInput, _caches, currentPos, _workspace);
                prevLogits.Dispose();

                // 5. Verify the draft prefix.
                //    verif_logits[0].argmax should equal draft[1] (head 0's prediction)
                //    verif_logits[1].argmax should equal draft[2] (head 1's prediction)
                //    ...
                //    verif_logits[K-1].argmax should equal draft[K] (head K-1's prediction)
                //
                //    token_0 is always accepted because it was the LM head's own
                //    greedy choice; verification begins at offset 0.
                //
                //    If a head's prediction disagrees with the model, all subsequent
                //    draft tokens are rejected.  This guarantees that the output is
                //    *identical* to greedy decoding — Medusa only changes throughput.
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

                // 6. Emit accepted tokens one-by-one (needed for streaming).
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
                    string fragment = _tokenizer.Decode(
                        scratchOne.AsSpan(0, 1), skipSpecials: true);
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
                if (generatedIds.Count > 0 &&
                    genCfg.StopTokenIds.Contains(generatedIds[^1]))
                    break;

                // 7. Prepare the starting state for the next round.
                //    Two cases depending on whether all K+1 draft tokens were accepted.
                if (accepted == draftLen)
                {
                    // All accepted: emit the bonus token
                    // verif_logits[K] predicts what comes after the last draft
                    // token (draft[K]).  We sample this greedily as the "bonus"
                    // — an un-verified token that we accept optimistically.
                    // (This is the standard speculative-decoding bonus step.)
                    int bonus = Sampler.Sample(
                        verifLogits.Data.Slice(numHeads * vocabSize, vocabSize),
                        SamplingConfig.Greedy, rng);
                    generatedIds.Add(bonus);
                    rateTracker.RecordToken();
                    TimeToFirstToken = rateTracker.TimeToFirstToken;
                    TokensPerSecond = rateTracker.RollingTokensPerSecond;
                    CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                    scratchOne[0] = bonus;
                    string bonusFragment = _tokenizer.Decode(
                        scratchOne.AsSpan(0, 1), skipSpecials: true);
                    decodedSoFar.Append(bonusFragment);

                    if (genCfg.Stream && bonusFragment.Length > 0)
                        yield return bonusFragment;

                    if (genCfg.StopTokenIds.Contains(bonus))
                        break;

                    // Sliding window: if the KV cache is full, keep only the
                    // last windowSize (or maxSeqLen/2) positions.
                    if (_caches[0].IsFull)
                    {
                        int keep = genCfg is { SlidingWindowSize: > 0 }
                            ? genCfg.SlidingWindowSize
                            : _caches[0].MaxSeqLen / 2;
                        for (int i = 0; i < _caches.Length; i++)
                            _caches[i].TrimToLast(keep);
                    }

                    currentPos += draftLen;

                    // The bonus token was sampled but never forwarded through
                    // the model.  We need a clean hidden + KV state for the
                    // next round, so we do a single ForwardLastLogits step.
                    // This extra pass slightly reduces the peak speedup, but
                    // it guarantees correct hidden state for the next Medusa
                    // prediction.
                    Tensor<float>? bonusPrev = verifLogits;
                    verifLogits = null;
                    _workspace.Reset();
                    using var bonusInput = _workspace.Rent<int>([1, 1]);
                    bonusInput.Data[0] = bonus;
                    logitsTensor = _model.ForwardLastLogits(
                        bonusInput, _caches, currentPos, _workspace);
                    bonusPrev.Dispose();
                    currentPos++;

                    // After single-token ForwardLastLogits, _cachedHidden is
                    // [1, 1, H] and already final-normed (ForwardInPlace).
                    // We copy directly into _normedHiddenScratch.
                    var ch = _model.LastCachedHidden;
                    ch?.Data[..hiddenDim].CopyTo(_normedHiddenScratch);
                }
                else
                {
                    // Partial acceptance: trim and re-extract state
                    int lastAcceptedIdx = accepted - 1;

                    // Remove the rejected suffix from every layer's KV cache.
                    // After Forward(draft), the cache was extended by draftLen
                    // positions; keeping only cacheLenBefore + accepted entries
                    // discards the unverified suffix.
                    for (int i = 0; i < _caches.Length; i++)
                        _caches[i].Truncate(cacheLenBefore + accepted);

                    // Sliding window check (same as above).
                    if (_caches[0].IsFull)
                    {
                        int keep = genCfg is { SlidingWindowSize: > 0 }
                            ? genCfg.SlidingWindowSize
                            : _caches[0].MaxSeqLen / 2;
                        for (int i = 0; i < _caches.Length; i++)
                            _caches[i].TrimToLast(keep);
                    }

                    // The next round's starting logits come from the last accepted
                    // position's verification logits.  These predict what token
                    // follows the last accepted token — exactly what we need to
                    // start the next draft.
                    //
                    // We MUST copy the data out of verifLogits (a workspace-rented
                    // tensor) because the workspace will be Reset on the next round.
                    var nextLogits = new Tensor<float>(1, vocabSize);
                    verifLogits.Data.Slice(
                        lastAcceptedIdx * vocabSize, vocabSize)
                        .CopyTo(nextLogits.Data);
                    logitsTensor = nextLogits;
                    verifLogits.Dispose();

                    // Re-norm the hidden state row for the last accepted position.
                    // _cachedHidden after Forward(draft) is [1, draftLen, H]
                    // pre-norm arch output.  We copy row (lastAcceptedIdx),
                    // apply final RMSNorm, and store in _normedHiddenScratch
                    // for the next Medusa prediction.
                    //
                    // Important correctness note: the hidden state at this row
                    // was computed with *only* the accepted prefix as context
                    // (causal masking prevents later positions from influencing
                    // earlier ones).  The rejected suffix does NOT pollute it.
                    RefillNormedHidden(lastAcceptedIdx);

                    currentPos += accepted;
                }
            }

            // Non-streaming: yield the full decoded text at the end.
            if (!genCfg.Stream)
            {
                string full = _tokenizer.Decode(
                    CollectionsMarshal.AsSpan(generatedIds), skipSpecials: true);
                if (full.Length > 0)
                    yield return full;
            }
        }
        finally
        {
            logitsTensor?.Dispose();
        }
    }

    // Helpers

    /// <summary>
    /// Copies row <paramref name="rowIndex"/> from the model's cached hidden
    /// state (Transformer.LastCachedHidden), applies the final RMSNorm, and
    /// stores the result into <see cref="_normedHiddenScratch"/>.
    ///
    /// This is needed because:
    ///   - After Forward(draft):  _cachedHidden is pre-norm arch output.
    ///   - After ForwardLastLogits (single-token): it is already normed.
    ///   - After prefill: it is pre-norm.
    ///
    /// We always re-norm because applying RMSNorm twice is extremely close
    /// to applying it once (the second norm's denominator ≈ 1), and the
    /// tiny deviation only affects Medusa-head accuracy (i.e. throughput),
    /// never output correctness (which is guaranteed by verification).
    /// </summary>
    private void RefillNormedHidden(int rowIndex)
    {
        var cachedHidden = _model.LastCachedHidden;
        if (cachedHidden == null) return;
        int hiddenDim = _model.Config.HiddenDim;

        // Extract the pre-norm row
        cachedHidden.Data.Slice(rowIndex * hiddenDim, hiddenDim)
            .CopyTo(_normedHiddenScratch);

        // Apply RMSNorm via a temp tensor (ForwardInPlace modifies in-place)
        using var normTemp = new Tensor<float>(1, hiddenDim);
        _normedHiddenScratch.CopyTo(normTemp.Data);
        _model.FinalNorm.ForwardInPlace(normTemp);
        normTemp.Data.CopyTo(_normedHiddenScratch);
    }

    // Cache management & disposal

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
        _medusaHeads.Dispose();
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Dispose();
        _workspace.Dispose();
    }

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(_disposed, nameof(MedusaGenerator<>));
}
