using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;

namespace SharpMind.Inference;

/// <summary>
/// Non-generic switchboard for <see cref="StandardGenerator{T}"/> host knobs so
/// callers don't need to name a concrete generic instantiation.
/// </summary>
public static class StandardGenerator
{
    /// <summary>
    /// When true, the decode loop awaits <see cref="Task.Yield"/> once per token so
    /// the host (e.g. single-threaded Blazor WASM) can repaint/process other work
    /// between steps. Off by default: desktop/server callers pay no overhead.
    /// </summary>
    public static bool YieldBetweenTokens { get; set; }
}

/// <summary>
/// Token generation loop for autoregressive decoding.
/// </summary>
public sealed class StandardGenerator<T> : IGenerator<T> where T : IKVCacheBuilder, new ()
{
    public string Name { get; init; } = "Standard";
    public IReadOnlyList<IKVCache> Caches => _caches;
    private readonly Transformer  _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly IKVCache[]     _caches;
    private readonly Random       _defaultRng;
    private readonly IWorkspace _workspace;
    /// <summary>Reused decode step (<c>[1]</c>) to avoid allocating a new <see cref="int"/>[] each token.</summary>
    private readonly int[]       _decodeTokenScratch = new int[1];
    /// <summary>Cached scratch buffer for repetition-penalty copy to avoid <see cref="ArrayPool{T}.Rent"/> per token.</summary>
    private float[]?              _penaltyScratch;
    /// <summary>Per-generation repetition-penalty bookkeeping.</summary>
    private readonly RepetitionPenaltyState _penaltyState = new();
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
    /// <summary>
    /// Absolute position of the next token to be fed. Cache length diverges from
    /// this after a sliding-window trim, so RoPE must use this, not
    /// <see cref="IKVCache.Length"/>.
    /// </summary>
    private int _position;

    /// <summary>Prompt-lookup logits rows of the pending run, [(draft + 1) x vocab]; allocated on first use.</summary>
    private float[]? _runRows;
    /// <summary>Prompt-lookup draft tokens of the pending run.</summary>
    private int[]? _runDraft;

    /// <summary>Prompt-lookup drafting counts for the last generation (all zero when it was off).</summary>
    public PromptLookupStats PromptLookupStats { get; private set; }

    /// <summary>MTP drafts proposed/accepted in the last generation (zero when it was off).</summary>
    public (int Drafted, int Accepted) MtpStats { get; private set; }

    // ── MTP (NextN) drafting ────────────────────────────────────────────────
    // A chain of up to MtpDraftTokens lookahead tokens from the model's draft
    // head. The head's cache only ever holds tokens that were actually fed to the
    // trunk plus the drafts of the round in flight, so a rejected draft truncates
    // it back to the verified prefix.
    private IKVCache? _mtpCache;
    private float[]?  _mtpHidden;        // trunk h of the last verified token
    private float[]?  _mtpChainHidden;   // head h after the last chain step (feeds the last draft)
    private float[]?  _mtpRunHidden;     // trunk h rows [0..m] of the verification run
    private Tensor<float>? _mtpHiddenTensor;
    private bool _mtpReady;              // _mtpHidden belongs to the token before the next one
    private bool _mtpDirty;              // drafted before its trunk forward completed
    private int _mtpDrafted, _mtpAccepted;
    /// <summary>MTP head cache length before the current draft chain.</summary>
    private int _mtpRunCacheLen;
    /// <summary>Recurrent-cache snapshots taken before the current draft run.</summary>
    private object?[]? _runSnapshots;
    /// <summary>The first token fed by the current draft run (the verified one).</summary>
    private int _runFirstToken;

    public StandardGenerator(
        Transformer   model,
        Tokenization.Tokenizer tokenizer,
        bool addBos, bool addEos,
        IKVCache[]?   caches = null,
        int?          seed = null,
        int?          maxCacheLen = null)
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
            int maxSeqLen = ModelConfig.ComputeMaxCacheLength(model.Config, maxCacheLen);
            int numKvHeads = model.Config.NumKvHeads;

            _caches = new IKVCache[numLayers];
            for (int i = 0; i < numLayers; i++)
                _caches[i] = model.Config.IsShortConvLayer(i)
                    ? new ShortConvCache(model.Config.ShortConvCacheLength - 1, model.Config.HiddenDim, 1, maxSeqLen)
                    : model.Config.IsRecurrentLayer(i)
                        ? new DeltaNetCache(model.Config.SsmConvKernel, model.Config.SsmConvDim,
                            model.Config.SsmTimeStepRank, model.Config.SsmStateSize,
                            model.Config.SsmValueHeadDim, 1, maxSeqLen)
                        : model.Config.IsMla
                            ? new T().CreateMlaKvCache(1, maxSeqLen,
                                model.Config.KvLoraRank + model.Config.MlaRopeHeadDim)
                            : new T().CreateKVCache(1, numKvHeads, maxSeqLen, model.Config.HeadDimForLayer(i));
        }

        _cacheTokens = _caches[0].Length == 0 ? [] : null;
        _position = _caches[0].Length;

        _defaultRng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        _workspace = MemoryHelpers.CreateWorkspace(Workspace.CalculateRequiredSize(model.Config.HiddenDim,model.Config.FfnDim,model.Config.VocabSize,model.Config.NumLayers, model.Config.MaxSeqLen));
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
        sampleCfg.Constraint?.Reset();

        if (promptIds.Length == 0)
            throw new InvalidOperationException("Prompt produced no token IDs; cannot generate.");

        // ── Tokens-per-second tracking (started before prefill to capture TTFT) ─
        var rateTracker = new TokenRateTracker(windowSize: 10);
        rateTracker.Start();

        Tensor<float>? logitsTensor = null;

        // Prompt-lookup drafting (see PromptLookupDrafter). A "run" is one batched forward over
        // [token, d1..dm]: it leaves m+1 logits rows and m unverified draft rows in the caches. Each
        // later step samples from row `runNext` exactly as it would sample from a fresh forward; a
        // token equal to the draft already in the cache needs no forward, the first different one
        // truncates the unverified rows away. Recurrent caches cannot be truncated: no drafting.
        int lookupMax = Math.Clamp(genCfg.PromptLookupDraft, 0, GenerationConfig.MaxPromptLookupDraft);
        if (lookupMax > 0 && _caches.Any(c => !c.SupportsTruncate)) lookupMax = 0;

        // MTP drafting needs the head. Models with recurrent layers are handled by
        // snapshotting their state around each draft run and replaying the verified
        // tokens on rejection (a rejected draft cannot rewind a recurrence). It also
        // takes precedence over prompt lookup (one draft source per step).
        bool mtp = genCfg.MtpDraft && _model.Mtp is not null;
        int mtpMax = mtp ? Math.Clamp(genCfg.MtpDraftTokens, 1, GenerationConfig.MaxMtpDraftTokens) : 0;
        // Recurrent caches cannot rewind; each draft run snapshots them and a
        // rejection restores the snapshot and replays the verified tokens.
        bool hasRecurrentCaches = _caches.Any(c => !c.SupportsTruncate);
        if (mtp)
        {
            lookupMax = 0;
            EnsureMtpCache();
            ResetMtpState();
            _mtpDrafted = 0;
            _mtpAccepted = 0;
            MtpStats = default;
        }

        PromptLookupDrafter? drafter = lookupMax > 0 ? new PromptLookupDrafter() : null;
        List<int>? lookupContext = null;
        int runLen = 0, runNext = 0, runCacheLen = 0, runPos = 0, lookupForwards = 0;
        PromptLookupStats = default;
        try
        {
                        // Prefill (chunked so long prompts fit the workspace; see Prefill)
            int posOffset = _position;
            // Someone else moved the cache under us: stop vouching for it.
            if (_cacheTokens is not null && _cacheTokens.Count != _caches[0].Length) _cacheTokens = null;
            // The drafter searches earlier turns too: take the vouched cache contents before the
            // capacity check below may stop vouching for them.
            if (drafter is not null)
                lookupContext = _cacheTokens is not null ? [.. _cacheTokens, .. promptIds] : [.. promptIds];
            // Prompt exceeds cache capacity — Prefill will trim mid-loop;
            // invalidate prefix tracking up-front.
            if (_cacheTokens is not null && _caches[0].Length + promptIds.Length > _caches[0].MaxSeqLen)
                _cacheTokens = null;
            logitsTensor = Prefill.ForwardLastLogitsChunked(
                _model, _caches, promptIds, _workspace, PrefillProgress, startPosition: posOffset);
            _position = posOffset + promptIds.Length;
            _cacheTokens?.AddRange(promptIds);
            if (drafter is not null)
            {
                int rows = (lookupMax + 1) * logitsTensor.Shape[1];
                if (_runRows is null || _runRows.Length < rows) _runRows = new float[rows];
                if (_runDraft is null || _runDraft.Length < lookupMax) _runDraft = new int[lookupMax];
            }
            else if (mtp)
            {
                // One verification forward covers the fed token plus every draft.
                if (_runRows is null || _runRows.Length < (1 + mtpMax) * logitsTensor.Shape[1])
                    _runRows = new float[(1 + mtpMax) * logitsTensor.Shape[1]];
                if (_runDraft is null || _runDraft.Length < mtpMax) _runDraft = new int[mtpMax];
            }


            int vocabSize = logitsTensor.Shape[1];
            int promptLen = promptIds.Length;

            _generatedIds = new List<int>(genCfg.MaxNewTokens);
            _penaltyState.Reset();
            var decodedSoFar = new System.Text.StringBuilder();
            // Incremental detokenizer window (see IncrementalDetokenizer).
            int detokPrefix = 0, detokRead = 0;
            // True when generation ended on a stop string/token, so held-back text
            // must not be flushed past it.
            bool endedOnStop = false;
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
            int maxPosition = _model.Config.MaxPosition;

            for (int step = 0; step < maxNewTokens; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // RoPE has no position past the declared context; stop rather than
                // throwing once the prompt plus generated tokens fill it.
                if (_position >= maxPosition) break;

                // Yield once per token when running on a single-threaded host
                // (e.g. Blazor WASM) so the browser can repaint between forward passes.
                if (StandardGenerator.YieldBetweenTokens) await Task.Yield();

                ReadOnlySpan<float> logitsSlice = runLen > 0
                    ? _runRows.AsSpan(runNext * vocabSize, vocabSize)
                    : logitsTensor!.Data[..vocabSize];
                GeneratorDiagnostics.PrintTopLogits(_tokenizer, step, logitsSlice);

                var tSample = DecodeProfiler.Begin();
                int nextId;
                if (repPenalty != 1.0f || sampleCfg.Constraint is not null)
                {
                    if (_penaltyScratch is null || _penaltyScratch.Length < vocabSize)
                        _penaltyScratch = new float[vocabSize];
                    Span<float> logits = _penaltyScratch.AsSpan(0, vocabSize);
                    logitsSlice.CopyTo(logits);
                    if (repPenalty != 1.0f)
                        _penaltyState.Apply(logits, promptIds, _generatedIds, repPenalty, repWindow);
                    if (!SamplingHelpers.TrySampleConstrained(logits, sampleCfg, rng, out nextId))
                    {
                        endedOnStop = true;
                        break;
                    }
                }
                else
                    nextId = Sampler.Sample(logitsSlice, sampleCfg, rng);

                _generatedIds.Add(nextId);
                lookupContext?.Add(nextId);
                OnTokenGenerated?.Invoke(nextId);

                rateTracker.RecordToken();
                TimeToFirstToken = rateTracker.TimeToFirstToken;
                TokensPerSecond = rateTracker.RollingTokensPerSecond;
                CumulativeTokensPerSecond = rateTracker.CumulativeTokensPerSecond;

                if (stopTokenIds.Contains(nextId)) { endedOnStop = true; break; }

                _decodeTokenScratch[0] = nextId;
                string fragment = IncrementalDetokenizer.Next(_tokenizer,
                    CollectionsMarshal.AsSpan(_generatedIds), ref detokPrefix, ref detokRead);
                decodedSoFar.Append(fragment);
                DecodeProfiler.Mark(DecodeStage.Sample, tSample);

                bool hitStop = false;
                int stopAt = int.MaxValue;
                if (maxStopLen > 0 && decodedSoFar.Length >= maxStopLen && _stopCheckBuf is not null)
                {
                    int start = decodedSoFar.Length - maxStopLen;
                    decodedSoFar.CopyTo(start, _stopCheckBuf, maxStopLen);
                    ReadOnlySpan<char> tail = _stopCheckBuf;
                    foreach (string stop in stopStrings)
                    {
                        int idx = tail.IndexOf(stop.AsSpan());
                        if (idx >= 0)
                        {
                            hitStop = true;
                            if (start + idx < stopAt) stopAt = start + idx;
                        }
                    }
                }

                if (hitStop)
                {
                    // Keep the fragment text that precedes the stop string; the stop
                    // marker itself may fall mid-fragment (e.g. "Hi END") and that
                    // whole prefix must not be swallowed with it.
                    int fragStart = decodedSoFar.Length - fragment.Length;
                    int keep = Math.Clamp(stopAt - fragStart, 0, fragment.Length);
                    fragment = keep < fragment.Length ? fragment[..keep] : fragment;
                    decodedSoFar.Length = fragStart + keep;
                }

                if (stream && fragment.Length > 0)
                {
                    // Buffer the fragment to yield outside the try-finally if needed, 
                    // but for simple Tensors, a try-finally without a catch is OK.
                    // Wait, actually yield return is allowed in try-finally, just not try-catch.
                    yield return fragment;
                }

                if (hitStop) { endedOnStop = true; break; }

                if (cache0.IsFull)
                {
                    int keep = slidingWindowSize > 0
                        ? slidingWindowSize
                        : cache0.MaxSeqLen / 2;
                    if (keep >= cache0.MaxSeqLen)
                        keep = Math.Max(1, cache0.MaxSeqLen / 2);
                    for (int i = 0; i < cachesLen; i++)
                        _caches[i].TrimToLast(keep);
                    // Entries now sit at positions they were not computed for;
                    // nothing about the cache is a prefix of any prompt any more.
                    _cacheTokens = null;
                    // The MTP head's cache also holds positions that just moved.
                    if (mtp) ResetMtpState();
                }

                if (runLen > 0)
                {
                    // The draft already in the cache: no forward, the next row is ready.
                    if (runNext < runLen && nextId == _runDraft![runNext])
                    {
                        runNext++;
                        _position++;
                        if (mtp)
                        {
                            _mtpAccepted++;
                            if (runNext == runLen)
                            {
                                // Every draft was accepted. The chain stopped one
                                // token short: its last step only predicted the last
                                // draft, so feed that token now with the chain's
                                // hidden state and mirror the trunk's cache.
                                FeedMtpHidden(_mtpChainHidden!, nextId, runPos + runLen);
                                AdoptMtpHiddenRow(runLen);
                                _mtpDirty = false;
                                _runSnapshots = null;
                            }
                        }
                        continue;
                    }

                    // Rejected draft on a model whose recurrent layers cannot rewind:
                    // restore their pre-run state and replay every verified token.
                    if (mtp && runNext < runLen && _runSnapshots is not null)
                    {
                        for (int i = 0; i < cachesLen; i++)
                        {
                            if (_caches[i].SupportsTruncate) _caches[i].Truncate(runCacheLen);
                            else if (_runSnapshots[i] is { } snapshot) _caches[i].Restore(snapshot);
                            else _caches[i].Reset();
                        }
                        if (_cacheTokens is not null && _cacheTokens.Count > runCacheLen)
                            _cacheTokens.RemoveRange(runCacheLen, _cacheTokens.Count - runCacheLen);

                        int replayLen = 1 + runNext;
                        _workspace.Reset();
                        using var replayInput = _workspace.Rent<int>([1, replayLen]);
                        replayInput.Data[0] = _runFirstToken;
                        _runDraft.AsSpan(0, runNext).CopyTo(replayInput.Data[1..]);
                        logitsTensor?.Dispose();
                        logitsTensor = _model.ForwardLastLogits(replayInput, _caches, runPos, _workspace);
                        _cacheTokens?.Add(_runFirstToken);
                        _cacheTokens?.AddRange(_runDraft.AsSpan(0, runNext));
                        _position = runPos + replayLen;
                        // The head's cache holds the whole chain; keep the tokens
                        // the trunk actually verified.
                        TruncateMtpCache(_mtpRunCacheLen + replayLen);
                        AdoptMtpHiddenRow(runNext);
                        _mtpDirty = false;
                        runLen = 0;
                        _runSnapshots = null;
                        // Fall through: this iteration still forwards `nextId`, which
                        // the trunk sampled after the last verified token.
                    }

                    // A different token drops the unverified rows; after the last row the run
                    // simply ends (every draft held, nextId is the extra token).
                    if (runNext < runLen)
                        RollBackRun(runCacheLen + 1 + runNext, runPos + 1 + runNext);
                    if (mtp && runNext < runLen)
                    {
                        // Keep the verified prefix in the head's cache too.
                        TruncateMtpCache(_mtpRunCacheLen + 1 + runNext);
                        AdoptMtpHiddenRow(runNext);
                        _mtpDirty = false;
                    }
                    if (drafter is not null) drafter.Record(runLen, runNext);
                    runLen = 0;
                    _runSnapshots = null;
                }

                Tensor<float>? prevTensor = logitsTensor;
                logitsTensor = null;
                int newPos = _position;

                _workspace.Reset();
                prevTensor?.Dispose();
                _decodeTokenScratch[0] = nextId;

                // Drafts must fit before the context end, before the cache is full (so no
                // sliding-window trim can land inside a run) and within the tokens still allowed.
                int m = 0;
                if (drafter is not null)
                {
                    int room = Math.Min(Math.Min(lookupMax, maxNewTokens - 1 - step),
                                        Math.Min(maxPosition - newPos - 1, cache0.MaxSeqLen - cache0.Length - 2));
                    if (room > 0) m = drafter.Propose(lookupContext!, _runDraft.AsSpan(0, room));
                    lookupForwards++;
                }
                else if (mtp && _mtpReady)
                {
                    int room = Math.Min(Math.Min(mtpMax, maxNewTokens - 1 - step),
                                        Math.Min(maxPosition - newPos - 1, cache0.MaxSeqLen - cache0.Length - 2));
                    if (_mtpCache is not null)
                        room = Math.Min(room, _mtpCache.MaxSeqLen - _mtpCache.Length - 1);
                    if (room >= 1)
                    {
                        _mtpRunCacheLen = _mtpCache!.Length;
                        int proposed = ProposeMtpDrafts(nextId, newPos, _runDraft.AsSpan(0, room));
                        if (proposed > 0)
                        {
                            m = proposed;
                            _mtpDrafted += proposed;
                            _mtpDirty = true;
                        }
                    }
                }

                if (m > 0 && mtp && hasRecurrentCaches)
                {
                    // Snapshot what cannot rewind, before the run consumes it.
                    for (int i = 0; i < cachesLen; i++)
                    {
                        if (_caches[i].SupportsTruncate) continue;
                        (_runSnapshots ??= new object?[cachesLen])[i] = _caches[i].Snapshot();
                    }
                }

                if (m == 0)
                {
                    using var stepInput = _workspace.Rent<int>([1, 1]);
                    stepInput.Data[0] = nextId;
                    logitsTensor = _model.ForwardLastLogits(stepInput, _caches, newPos, _workspace);
                    _cacheTokens?.Add(nextId);
                    if (mtp) CaptureMtpHidden();
                }
                else
                {
                    runCacheLen = cache0.Length;
                    runPos = newPos;
                    _runFirstToken = nextId;
                    using var runInput = _workspace.Rent<int>([1, 1 + m]);
                    runInput.Data[0] = nextId;
                    _runDraft.AsSpan(0, m).CopyTo(runInput.Data[1..]);
                    using (var allLogits = _model.Forward(runInput, _caches, newPos, _workspace))
                        allLogits.Data[..((1 + m) * vocabSize)].CopyTo(_runRows);
                    runLen = m;
                    runNext = 0;
                    _cacheTokens?.Add(nextId);
                    _cacheTokens?.AddRange(_runDraft.AsSpan(0, m));
                    if (mtp) CaptureMtpHiddenRows(m);
                }
                _position++;
            }

            if (!genCfg.Stream)
                yield return _tokenizer.Decode(CollectionsMarshal.AsSpan(_generatedIds), skipSpecials: true).Replace("\uFFFD", "");
            else if (!endedOnStop)
            {
                // Emit any text still held back for an incomplete sequence at the end.
                string tail = IncrementalDetokenizer.Next(_tokenizer,
                    CollectionsMarshal.AsSpan(_generatedIds), ref detokPrefix, ref detokRead, final: true);
                if (tail.Length > 0) yield return tail;
            }
        }
        finally
        {
            // Left inside a run (stop, token limit, a consumer that stopped reading, cancellation):
            // keep only the verified rows, so the next turn's prefix reuse sees what was emitted.
            if (runLen > 0)
            {
                if (_runSnapshots is not null && runNext < runLen)
                {
                    // Stopped inside a run that a recurrence cannot rewind: drop the
                    // whole run back to the pre-run state and stop vouching for the
                    // cache contents this turn.
                    for (int i = 0; i < _caches.Length; i++)
                    {
                        if (_caches[i].SupportsTruncate) _caches[i].Truncate(runCacheLen);
                        else if (_runSnapshots[i] is { } snapshot) _caches[i].Restore(snapshot);
                        else _caches[i].Reset();
                    }
                    _cacheTokens = null;
                    _position = runCacheLen;
                    if (mtp)
                    {
                        // The trunk kept nothing from the run; so does the head.
                        TruncateMtpCache(_mtpRunCacheLen);
                        _mtpReady = false;
                        _mtpDirty = false;
                    }
                }
                else
                {
                    if (runNext < runLen)
                    {
                        RollBackRun(runCacheLen + 1 + runNext, runPos + 1 + runNext);
                        if (mtp)
                        {
                            TruncateMtpCache(_mtpRunCacheLen + 1 + runNext);
                            AdoptMtpHiddenRow(runNext);
                            _mtpDirty = false;
                        }
                    }
                    if (drafter is not null) drafter.Record(runLen, runNext);
                }
                _runSnapshots = null;
            }
            if (drafter is not null)
                PromptLookupStats = new PromptLookupStats(lookupForwards, drafter.Drafted, drafter.Accepted);
            if (mtp)
            {
                // A draft proposed but never verified leaves the head's cache one
                // token ahead of the trunk; drop it rather than reuse it.
                if (_mtpDirty) ResetMtpState();
                MtpStats = (_mtpDrafted, _mtpAccepted);
            }
            logitsTensor?.Dispose();
        }
    }

    private void EnsureMtpCache()
    {
        if (_mtpCache is not null) return;
        var cfg = _model.Config;
        int length = _caches.Length > 0 ? _caches[0].MaxSeqLen : cfg.MaxSeqLen;
        _mtpCache = cfg.IsMla
            ? new MlaKvCache(1, length, cfg.KvLoraRank + cfg.MlaRopeHeadDim)
            : new KVCache(1, cfg.NumKvHeads, length, cfg.HeadDim);
    }

    /// <summary>
    /// Forgets the draft head's history. Called whenever the trunk's positions
    /// change under it (reset, truncate, sliding-window trim) and after a draft
    /// that never reached the trunk.
    /// </summary>
    private void ResetMtpState()
    {
        _mtpCache?.Reset();
        _mtpReady = false;
        _mtpDirty = false;
    }

    private void AdoptMtpHiddenRow(int row)
    {
        _mtpReady = false;
        int n = _model.Config.HiddenDim;
        if (_mtpRunHidden is null || row < 0 || (long)(row + 1) * n > _mtpRunHidden.Length) return;
        if (_mtpHidden is null || _mtpHidden.Length < n) _mtpHidden = new float[n];
        _mtpRunHidden.AsSpan(row * n, n).CopyTo(_mtpHidden);
        _mtpReady = true;
    }

    /// <summary>Stores the hidden state of the single token just fed.</summary>
    private void CaptureMtpHidden()
    {
        var normed = _model.LastNormed;
        if (normed is null) { _mtpReady = false; return; }
        int n = _model.Config.HiddenDim;
        if (_mtpHidden is null) _mtpHidden = new float[n];
        normed.Data[..n].CopyTo(_mtpHidden);
        _mtpReady = true;
        _mtpDirty = false;
    }

    /// <summary>Stores the hidden state of every row of a verification forward.</summary>
    private void CaptureMtpHiddenRows(int m)
    {
        int n = _model.Config.HiddenDim;
        int rows = m + 1;
        if (_mtpRunHidden is null || _mtpRunHidden.Length < rows * n)
            _mtpRunHidden = new float[rows * n];

        for (int i = 0; i < rows; i++)
        {
            using var row = _model.GetNormedHiddenRow(i);
            if (row is null) { _mtpReady = false; return; }
            row.Data[..n].CopyTo(_mtpRunHidden.AsSpan(i * n, n));
        }

        // The rows are only adopted when the run resolves; until then no draft
        // may be built from them.
        _mtpReady = false;
    }

    /// <summary>
    /// Asks the MTP head for a chain of draft tokens after <paramref name="token"/>.
    /// Each step feeds the previous step's head-normed hidden state and the token
    /// it predicted, so one verification forward covers the whole chain. The head
    /// appends every fed token to its own cache; a rejected chain is truncated back
    /// to the verified prefix by the caller.
    /// </summary>
    /// <returns>The number of drafts written to <paramref name="drafts"/>.</returns>
    private int ProposeMtpDrafts(int token, int position, Span<int> drafts)
    {
        if (!_mtpReady || _mtpHidden is null || _mtpCache is null || drafts.Length == 0) return 0;

        var cfg = _model.Config;
        int hidden = cfg.HiddenDim;
        int vocab = cfg.VocabSize;
        _mtpHiddenTensor ??= new Tensor<float>(1, 1, hidden);
        _mtpChainHidden ??= new float[hidden];

        _mtpHidden.AsSpan(0, hidden).CopyTo(_mtpHiddenTensor.Data);

        using var ids = new Tensor<int>(1, 1);
        int count = 0;
        for (int j = 0; j < drafts.Length; j++)
        {
            ids.Data[0] = token;
            var (nextHidden, logits) = _model.ForwardMtp(
                _mtpHiddenTensor, ids, _mtpCache, position + j);
            try
            {
                int draft = ArgMax(logits.Data[..vocab]);
                drafts[j] = draft;
                count++;

                // The head's hidden for `token` is the next step's input.
                nextHidden.Data[..hidden].CopyTo(_mtpChainHidden);
                token = draft;
                _mtpChainHidden.AsSpan(0, hidden).CopyTo(_mtpHiddenTensor.Data);
            }
            finally
            {
                nextHidden.Dispose();
                logits.Dispose();
            }
        }

        return count;
    }

    /// <summary>
    /// Appends <paramref name="token"/> to the head's cache without projecting the
    /// LM head. An accepted draft was fed to the trunk, so the head must see it too
    /// — otherwise the next draft attends over a stream with that token missing.
    /// </summary>
    private void FeedMtpHidden(float[] hidden, int token, int position)
    {
        if (_mtpCache is null) return;

        var cfg = _model.Config;
        int n = cfg.HiddenDim;
        _mtpHiddenTensor ??= new Tensor<float>(1, 1, n);
        hidden.AsSpan(0, n).CopyTo(_mtpHiddenTensor.Data);

        using var ids = new Tensor<int>(1, 1);
        ids.Data[0] = token;
        using var _ = _model.ForwardMtpHidden(_mtpHiddenTensor, ids, _mtpCache, position);
    }

    /// <summary>Drops the head's unverified chain entries, keeping a verified prefix.</summary>
    private void TruncateMtpCache(int length)
    {
        if (_mtpCache is null) return;
        length = Math.Clamp(length, 0, _mtpCache.Length);
        if (_mtpCache.Length > length) _mtpCache.Truncate(length);
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        float max = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > max) { max = values[i]; best = i; }
        return best;
    }

    /// <summary>Drops the unverified draft rows of a prompt-lookup run.</summary>
    private void RollBackRun(int keepLen, int position)
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Truncate(keepLen);
        if (_cacheTokens is not null && _cacheTokens.Count > keepLen)
            _cacheTokens.RemoveRange(keepLen, _cacheTokens.Count - keepLen);
        _position = position;
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
        _position = 0;
        ResetMtpState();
    }

    public IReadOnlyList<int>? CacheTokens => _cacheTokens;

    public void TruncateCache(int length)
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].Truncate(length);
        if (_cacheTokens is not null && _cacheTokens.Count > length)
            _cacheTokens.RemoveRange(length, _cacheTokens.Count - length);
        _position = length;
        // The head's cache cannot be rewound to an arbitrary trunk prefix without
        // replaying it; start it fresh on the next turn.
        ResetMtpState();
    }

    public void SetCacheTokens(IReadOnlyList<int> tokens)
    {
        _cacheTokens = [.. tokens];
        _position = tokens.Count;
        ResetMtpState();
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
        _mtpCache?.Dispose();
        _mtpHiddenTensor?.Dispose();
        _workspace.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(StandardGenerator<>));
}

