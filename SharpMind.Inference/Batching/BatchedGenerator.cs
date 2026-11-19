using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading.Channels;
using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;

namespace SharpMind.Inference;

/// <summary>
/// Continuous-batching token generator: one model, one set of batched KV caches,
/// many sequences.
///
/// <see cref="StandardGenerator{T}"/> owns one sequence per model — a server
/// runs N of them and pays N forwards per token. Here every active sequence
/// advances by one token in a single forward pass, which is what makes batch
/// throughput possible on a CPU: the decode step is dominated by reading the
/// weight matrices, and that cost is shared by all rows.
///
/// Scheduling per iteration:
/// <list type="bullet">
///   <item>admitting queued requests into free rows,</item>
///   <item>one prefill chunk for each sequence still reading its prompt,</item>
///   <item>one batched decode step for every sequence that has a token ready,</item>
///   <item>reaping finished/cancelled sequences and compacting their rows.</item>
/// </list>
///
/// Prompts may have different lengths and sequences may sit at different
/// positions: each cache row carries its own length (<see cref="IKVCache.LengthForRow"/>)
/// and the forward is given a <see cref="KVCacheRowView"/> so each row's rotary
/// positions and attention window come from that row's own history.
///
/// Per-request output is a stream of decoded fragments plus
/// <see cref="BatchedSequence.Text"/>/<see cref="BatchedSequence.FinishReason"/>.
/// For a dense model, greedy sampling produces exactly the tokens each request
/// would get from <see cref="StandardGenerator{T}"/> alone (pinned by test).
/// A routed (MoE) model evaluates the same experts with the same weights, but
/// grouping several rows into one matmul can round differently in the last bit,
/// so a near-tie in the logits may break the other way.
/// </summary>
public sealed class BatchedGenerator : IDisposable, IAsyncDisposable
{
    private readonly Transformer _model;
    private readonly Tokenization.Tokenizer _tokenizer;
    private readonly BatchedGeneratorOptions _options;
    private readonly IKVCache[] _caches;
    private readonly KVCacheRowView[] _views;
    private readonly IKVCache? _mtpCache;
    private readonly KVCacheRowView? _mtpView;
    private readonly int _mtpMaxTokens;
    private readonly IWorkspace _workspace;
    private readonly Channel<BatchedSequence> _inbox;
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Task _loop;
    private readonly int _vocab;
    private readonly int _seqLen;
    private readonly int _maxPosition;
    /// <summary>Chunk size actually used; bounded by the workspace's prefill budget.</summary>
    private readonly int _maxChunkLength;
    private readonly int[] _rowSlots;
    private readonly BatchedSequence[] _rowSeqs;
    private readonly List<BatchedSequence> _pending = [];
    private readonly List<BatchedSequence> _active = [];
    private int _nextId;
    private int _outstanding;
    private bool _disposed;
    private Exception? _fatal;

    // Aggregate decode statistics.
    private long _steps;
    private long _decodedTokens;
    private double _decodeSeconds;
    private long _batchTotal;

    public BatchedGenerator(
        Transformer model,
        Tokenization.Tokenizer tokenizer,
        BatchedGeneratorOptions? options = null,
        IKVCacheBuilder? cacheBuilder = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);

        _model = model;
        _tokenizer = tokenizer;
        _options = options ?? new BatchedGeneratorOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxBatch, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxQueued, 0);
        ArgumentOutOfRangeException.ThrowIfLessThan(_options.MaxChunkLength, 1);

        var config = model.Config;
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (config.IsShortConvLayer(i) || config.IsRecurrentLayer(i))
                throw new NotSupportedException(
                    "BatchedGenerator requires attention-only models: layer " + i +
                    " is recurrent and its state cannot be addressed per row.");
        }

        _seqLen = ModelConfig.ComputeMaxCacheLength(config, _options.MaxSeqLen);
        _maxPosition = config.MaxPosition;
        _vocab = config.VocabSize;
        // The workspace budgets for min(MaxSeqLen, 128) prefill tokens; a larger
        // chunk would overflow its bump allocator.
        _maxChunkLength = Math.Max(1, Math.Min(_options.MaxChunkLength, Math.Min(_seqLen, 128)));

        var builder = cacheBuilder ?? new KVCacherBuilder();
        _caches = new IKVCache[config.NumLayers];
        _views = new KVCacheRowView[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            var cache = config.IsMla
                ? builder.CreateMlaKvCache(_options.MaxBatch, _seqLen, config.KvLoraRank + config.MlaRopeHeadDim)
                : builder.CreateKVCache(_options.MaxBatch, config.NumKvHeads, _seqLen, config.HeadDimForLayer(i));
            if (!cache.SupportsRaggedRows)
                throw new NotSupportedException(
                    $"BatchedGenerator needs a cache that tracks per-row lengths; " +
                    $"{builder.GetType().Name} produced {cache.GetType().Name}.");
            _caches[i] = cache;
            _views[i] = new KVCacheRowView(cache);
        }

        if (model.Mtp is not null)
        {
            var mtpCache = config.IsMla
                ? builder.CreateMlaKvCache(_options.MaxBatch, _seqLen, config.KvLoraRank + config.MlaRopeHeadDim)
                : builder.CreateKVCache(_options.MaxBatch, config.NumKvHeads, _seqLen, config.HeadDim);
            if (!mtpCache.SupportsRaggedRows)
                throw new NotSupportedException(
                    $"BatchedGenerator needs a per-row MTP cache; " +
                    $"{builder.GetType().Name} produced {mtpCache.GetType().Name}.");
            _mtpCache = mtpCache;
            _mtpView = new KVCacheRowView(mtpCache);
            _mtpMaxTokens = GenerationConfig.MaxMtpDraftTokens;
        }

        _workspace = MemoryHelpers.CreateWorkspace(Workspace.CalculateRequiredSize(
            config.HiddenDim, config.FfnDim, config.VocabSize, config.NumLayers, config.MaxSeqLen));

        _rowSlots = new int[_options.MaxBatch];
        _rowSeqs = new BatchedSequence[_options.MaxBatch];
        _inbox = Channel.CreateUnbounded<BatchedSequence>(new UnboundedChannelOptions
        {
            SingleReader = true,
            SingleWriter = false,
            AllowSynchronousContinuations = false,
        });

        _loop = Task.Factory.StartNew(RunLoop, CancellationToken.None,
            TaskCreationOptions.LongRunning, TaskScheduler.Default);
    }

    /// <summary>Batch width this generator was built for.</summary>
    public int MaxBatch => _options.MaxBatch;

    /// <summary>Tokens one row can hold; the per-request context budget.</summary>
    public int MaxSequenceLength => _seqLen;

    /// <summary>
    /// Optional lease acquired once per forward pass and disposed when the pass
    /// ends. A host that shares one <see cref="Transformer"/> between engines
    /// (chat sessions and this batcher) must serialise access: the transformer
    /// keeps its activations in instance fields, so two concurrent forwards
    /// corrupt each other. The gate is what makes batching and chat safe on one
    /// loaded model — the batcher holds the lease only for the forward itself.
    /// </summary>
    public Func<CancellationToken, ValueTask<IDisposable>>? StepGate { get; set; }

    /// <summary>Rows currently decoding or prefilling.</summary>
    public int Active => _active.Count;

    /// <summary>Requests that have finished and released their row.</summary>
    public long SequencesCompleted => Interlocked.Read(ref _sequencesCompleted);

    private long _sequencesCompleted;

    /// <summary>Aggregate throughput, updated after every decode step.</summary>
    public BatchedGeneratorStats Stats
    {
        get
        {
            long steps = Interlocked.Read(ref _steps);
            long tokens = Interlocked.Read(ref _decodedTokens);
            double seconds = Volatile.Read(ref _decodeSeconds);
            return new(steps, tokens, seconds,
                seconds > 0 ? tokens / seconds : 0,
                steps > 0 ? (double)Interlocked.Read(ref _batchTotal) / steps : 0);
        }
    }

    /// <summary>The per-layer KV caches. Exposed for diagnostics/tests.</summary>
    public IReadOnlyList<IKVCache> Caches => _caches;

    // ── Public API ───────────────────────────────────────────────────────────

    /// <summary>Tokenises <paramref name="prompt"/> and enqueues it.</summary>
    public ValueTask<BatchedSequence> EnqueueAsync(
        string prompt,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default)
        => EnqueueTokensAsync(_tokenizer.Encode(prompt, _options.AddBos, _options.AddEos),
            sampling, generation, cancellationToken);

    /// <summary>
    /// Enqueues already-encoded prompt tokens. The returned handle starts
    /// streaming as soon as the scheduler admits it; awaiting the call only
    /// reserves queue space, it does not wait for a row.
    /// </summary>
    public async ValueTask<BatchedSequence> EnqueueTokensAsync(
        int[] promptIds,
        SamplingConfig? sampling = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(promptIds);
        if (promptIds.Length == 0)
            throw new ArgumentException("Prompt produced no token IDs; cannot generate.", nameof(promptIds));
        if (promptIds.Length > _seqLen)
            throw new ArgumentException(
                $"Prompt of {promptIds.Length} tokens exceeds the cache capacity of {_seqLen}.", nameof(promptIds));
        if (promptIds.Length > _maxPosition)
            throw new ArgumentException(
                $"Prompt of {promptIds.Length} tokens exceeds the model's context of {_maxPosition}.", nameof(promptIds));

        if (_fatal is not null)
            throw new InvalidOperationException("The batched generator has stopped after an error.", _fatal);

        int outstanding = Interlocked.Increment(ref _outstanding);
        if (outstanding > _options.MaxBatch + _options.MaxQueued)
        {
            Interlocked.Decrement(ref _outstanding);
            throw new BatchQueueFullException(_options.MaxBatch, _options.MaxQueued);
        }

        var sampleCfg = sampling ?? SamplingConfig.Greedy;
        var genCfg = generation ?? GenerationConfig.Default;
        sampleCfg.Constraint?.Reset();

        var seq = new BatchedSequence(Interlocked.Increment(ref _nextId), promptIds,
            sampleCfg, genCfg, cancellationToken);
        seq.Rng = sampleCfg.Seed.HasValue ? new Random(sampleCfg.Seed.Value)
                 : _options.Seed.HasValue ? new Random(_options.Seed.Value + seq.Id)
                 : Random.Shared;

        try
        {
            await _inbox.Writer.WriteAsync(seq, cancellationToken).ConfigureAwait(false);
        }
        catch
        {
            Interlocked.Decrement(ref _outstanding);
            seq.MarkReleased();
            await seq.DisposeAsync().ConfigureAwait(false);
            throw;
        }

        return seq;
    }

    // ── Scheduler ────────────────────────────────────────────────────────────

    private void RunLoop()
    {
        try
        {
            while (!_shutdown.IsCancellationRequested)
            {
                DrainInbox();
                Reap();

                if (_active.Count == 0 && _pending.Count == 0)
                {
                    if (!_inbox.Reader.WaitToReadAsync(_shutdown.Token).AsTask().GetAwaiter().GetResult())
                        break;
                    continue;
                }

                Admit();
                Step();
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            // A model/kernel failure is not recoverable per sequence; stop the
            // loop and surface it to everyone in flight.
            _fatal = ex;
        }
        finally
        {
            _inbox.Writer.TryComplete(_fatal);
            foreach (var seq in _active) { CancelRemaining(seq); seq.MarkReleased(); }
            foreach (var seq in _pending) { CancelRemaining(seq); seq.MarkReleased(); }
            _active.Clear();
            _pending.Clear();
        }
    }

    private void DrainInbox()
    {
        while (_inbox.Reader.TryRead(out var seq))
            _pending.Add(seq);
    }

    private void Admit()
    {
        while (_pending.Count > 0 && _active.Count < _options.MaxBatch)
        {
            var seq = _pending[0];
            _pending.RemoveAt(0);

            if (seq.IsCompleted || seq.AbandonToken.IsCancellationRequested)
            {
                Finish(seq, BatchFinishReason.Cancelled);
                seq.MarkReleased();
                Interlocked.Decrement(ref _outstanding);
                continue;
            }

            seq.Slot = _active.Count;
            seq.State = SequenceState.Prefilling;
            seq.PromptFed = 0;
            seq.MtpReady = false;
            seq.MtpPrefillPrev = null;
            _mtpCache?.ClearRow(seq.Slot);
            _active.Add(seq);
        }
    }

    /// <summary>Drops finished and abandoned sequences, keeping rows dense.</summary>
    private void Reap()
    {
        for (int i = _active.Count - 1; i >= 0; i--)
        {
            var seq = _active[i];
            if (seq.State != SequenceState.Finished && !seq.AbandonToken.IsCancellationRequested)
                continue;

            if (seq.State != SequenceState.Finished)
                Finish(seq, BatchFinishReason.Cancelled);

            RemoveAt(i);
        }

        for (int i = _pending.Count - 1; i >= 0; i--)
        {
            var seq = _pending[i];
            if (!seq.AbandonToken.IsCancellationRequested && !seq.IsCompleted) continue;
            _pending.RemoveAt(i);
            Finish(seq, BatchFinishReason.Cancelled);
            seq.MarkReleased();
            Interlocked.Decrement(ref _outstanding);
        }
    }

    private void RemoveAt(int index)
    {
        var removed = _active[index];
        int last = _active.Count - 1;
        if (index != last)
        {
            for (int i = 0; i < _caches.Length; i++)
                _caches[i].MoveRow(last, index);
            _mtpCache?.MoveRow(last, index);
            var moved = _active[last];
            moved.Slot = index;
            _active[index] = moved;
        }
        else
        {
            // Nobody moves into this row, so its length must be cleared or the
            // next sequence admitted here would append after stale positions.
            for (int i = 0; i < _caches.Length; i++)
                _caches[i].ClearRow(index);
            _mtpCache?.ClearRow(index);
        }
        _active.RemoveAt(last);
        removed.MarkReleased();
        Interlocked.Decrement(ref _outstanding);
        Interlocked.Increment(ref _sequencesCompleted);
    }

    private void Step()
    {
        // One chunk per prefilling sequence so a long prompt cannot starve the
        // sequences that already sit in the decode batch.
        for (int i = 0; i < _active.Count; i++)
        {
            var seq = _active[i];
            if (seq.State == SequenceState.Prefilling)
                PrefillChunk(seq);
        }

        DecodeStep();
    }

    private void PrefillChunk(BatchedSequence seq)
    {
        if (seq.AbandonToken.IsCancellationRequested) { Finish(seq, BatchFinishReason.Cancelled); return; }

        int remaining = seq.PromptTokens.Length - seq.PromptFed;
        int room = _seqLen - _caches[0].LengthForRow(seq.Slot);
        int len = Math.Min(Math.Min(_maxChunkLength, remaining), room);
        if (len <= 0) { Finish(seq, BatchFinishReason.ContextFull); return; }

        bool last = len == remaining;
        int startPosition = _caches[0].LengthForRow(seq.Slot);
        bool mtpPrefill = _mtpCache is not null && seq.Generation.MtpDraft;

        _rowSlots[0] = seq.Slot;
        for (int i = 0; i < _views.Length; i++) _views[i].SetRows(_rowSlots.AsSpan(0, 1));

        _workspace.Reset();
        using var input = _workspace.Rent<int>([1, len]);
        seq.PromptTokens.AsSpan(seq.PromptFed, len).CopyTo(input.Data);

        if (!last)
        {
            using (AcquireStepGate()) _model.ForwardFill(input, _views, 0, _workspace);
            if (mtpPrefill) FeedMtpPrefillChunk(seq, input, startPosition);
            seq.PromptFed += len;
            // Fired off the caller's thread (the scheduler's), so a host callback
            // must be thread-safe. ChatSession queues these on a concurrent queue.
            seq.PrefillProgress?.Invoke((double)seq.PromptFed / seq.PromptTokens.Length);
            return;
        }

        Tensor<float> logits;
        using (AcquireStepGate())
            logits = _model.ForwardLastLogits(input, _views, 0, _workspace);

        using (logits)
        {
            if (mtpPrefill) FeedMtpPrefillChunk(seq, input, startPosition);
            seq.PromptFed += len;
            SampleAndEmit(seq, logits.Data[.._vocab]);
        }
    }

    /// <summary>Takes the host's model lease for one forward pass, when one is configured.</summary>
    private IDisposable? AcquireStepGate()
        => StepGate?.Invoke(_shutdown.Token).AsTask().GetAwaiter().GetResult();

    private void DecodeStep()
    {
        int count = 0;
        for (int i = 0; i < _active.Count; i++)
        {
            var seq = _active[i];
            if (seq.State != SequenceState.Decoding) continue;
            _rowSlots[count] = seq.Slot;
            _rowSeqs[count] = seq;
            count++;
        }
        if (count == 0) return;

        if (_mtpCache is not null && TryDecodeMtp(count)) return;
        NormalDecodeStep(count);
    }

    private void NormalDecodeStep(int count)
    {
        for (int i = 0; i < _views.Length; i++) _views[i].SetRows(_rowSlots.AsSpan(0, count));

        _workspace.Reset();
        using var input = _workspace.Rent<int>([count, 1]);
        for (int i = 0; i < count; i++) input.Data[i] = _rowSeqs[i].PendingToken;

        var sw = Stopwatch.GetTimestamp();
        Tensor<float> logits;
        using (AcquireStepGate())
            logits = _model.ForwardLastLogits(input, _views, 0, _workspace);

        // Publish the step's counters before any sequence can complete, so a
        // caller that awaits a sequence and then reads Stats sees its tokens.
        double seconds = Stopwatch.GetElapsedTime(sw).TotalSeconds;
        Interlocked.Add(ref _steps, 1);
        Interlocked.Add(ref _batchTotal, count);
        Interlocked.Add(ref _decodedTokens, count);
        double total = Volatile.Read(ref _decodeSeconds) + seconds;
        Volatile.Write(ref _decodeSeconds, total);

        using (logits)
        {
            if (_mtpCache is not null)
                CaptureMtpHiddenAfterDecode(count);

            var data = logits.Data;
            for (int i = 0; i < count; i++)
                SampleAndEmit(_rowSeqs[i], data.Slice(i * _vocab, _vocab));
        }
    }

    /// <summary>
    /// One batched MTP round: chain drafts for every active row, verify them in a
    /// single <c>[count, 1+k]</c> forward, then accept/reject per row. Returns
    /// false when any row cannot draft this step, so the caller falls back to the
    /// ordinary one-token decode.
    /// </summary>
    private bool TryDecodeMtp(int count)
    {
        var cfg = _model.Config;
        int hidden = cfg.HiddenDim;
        int k = int.MaxValue;

        for (int i = 0; i < count; i++)
        {
            var seq = _rowSeqs[i];
            if (!seq.Generation.MtpDraft || !seq.MtpReady || seq.MtpHidden is null)
                return false;
            if (_mtpCache is null || _mtpView is null) return false;

            int trunkLen = _caches[0].LengthForRow(seq.Slot);
            int mtpLen = _mtpCache.LengthForRow(seq.Slot);
            int room = Math.Min(Math.Clamp(seq.Generation.MtpDraftTokens, 1, _mtpMaxTokens),
                Math.Min(seq.Generation.MaxNewTokens - seq.Generated.Count - 1,
                    Math.Min(_maxPosition - trunkLen - 1,
                        Math.Min(_seqLen - trunkLen - 2, _seqLen - mtpLen - 1))));
            if (room < 1) return false;
            if (room < k) k = room;
        }

        if (k < 1) return false;

        for (int i = 0; i < count; i++)
        {
            var seq = _rowSeqs[i];
            seq.MtpTrunkLenBeforeRun = _caches[0].LengthForRow(seq.Slot);
            seq.MtpLenBeforeRun = _mtpCache!.LengthForRow(seq.Slot);
            seq.MtpDraftCount = k;
            if (seq.MtpDrafts is null || seq.MtpDrafts.Length < k) seq.MtpDrafts = new int[k];
            if (seq.MtpChainHidden is null || seq.MtpChainHidden.Length < hidden)
                seq.MtpChainHidden = new float[hidden];
            seq.MtpDrafted += k;
        }

        _mtpView!.SetRows(_rowSlots.AsSpan(0, count));
        Span<int> positions = count <= 64 ? stackalloc int[count] : new int[count];

        using var hTensor = new Tensor<float>(count, 1, hidden);
        using var idsTensor = new Tensor<int>(count, 1);

        // Chain k draft steps. Each step feeds the previous step's head-normed
        // hidden with the token it predicted, exactly like the single-sequence
        // round; the view supplies each row's trunk position for RoPE while the
        // head cache itself stays append-ordered.
        for (int step = 0; step < k; step++)
        {
            for (int i = 0; i < count; i++)
            {
                var seq = _rowSeqs[i];
                var source = step == 0 ? seq.MtpHidden! : seq.MtpChainHidden!;
                source.AsSpan(0, hidden).CopyTo(hTensor.Data.Slice(i * hidden, hidden));
                idsTensor.Data[i] = step == 0 ? seq.PendingToken : seq.MtpDrafts![step - 1];
                positions[i] = seq.MtpTrunkLenBeforeRun + step;
            }

            _mtpView.SetPositions(positions);
            _workspace.Reset();
            Tensor<float> nextHidden;
            Tensor<float> logits;
            using (AcquireStepGate())
                (nextHidden, logits) = _model.ForwardMtp(hTensor, idsTensor, _mtpView, 0, _workspace);
            try
            {
                for (int i = 0; i < count; i++)
                {
                    var seq = _rowSeqs[i];
                    int draft = ArgMax(logits.Data.Slice(i * _vocab, _vocab));
                    seq.MtpDrafts![step] = draft;
                    nextHidden.Data.Slice(i * hidden, hidden).CopyTo(seq.MtpChainHidden!);
                }
            }
            finally
            {
                nextHidden.Dispose();
                logits.Dispose();
            }

            Interlocked.Add(ref _steps, 1);
        }

        // One verification forward covers every row's pending token and drafts.
        _workspace.Reset();
        using var runInput = _workspace.Rent<int>([count, 1 + k]);
        for (int i = 0; i < count; i++)
        {
            var seq = _rowSeqs[i];
            int offset = i * (1 + k);
            runInput.Data[offset] = seq.PendingToken;
            for (int j = 0; j < k; j++)
                runInput.Data[offset + 1 + j] = seq.MtpDrafts![j];
        }

        for (int i = 0; i < _views.Length; i++) _views[i].SetRows(_rowSlots.AsSpan(0, count));

        var sw = Stopwatch.GetTimestamp();
        Tensor<float> allLogits;
        using (AcquireStepGate())
            allLogits = _model.Forward(runInput, _views, 0, _workspace);

        double seconds = Stopwatch.GetElapsedTime(sw).TotalSeconds;
        Interlocked.Add(ref _steps, 1);
        Interlocked.Add(ref _batchTotal, count);
        double total = Volatile.Read(ref _decodeSeconds) + seconds;
        Volatile.Write(ref _decodeSeconds, total);

        int emitted = 0;
        using (allLogits)
        {
            for (int i = 0; i < count; i++)
                CaptureMtpRunRows(_rowSeqs[i], i * (1 + k), k);

            for (int i = 0; i < count; i++)
            {
                var seq = _rowSeqs[i];
                int trunkLen = seq.MtpTrunkLenBeforeRun;
                int mtpLen = seq.MtpLenBeforeRun;
                int runNext = 0;
                bool stopped = false;

                for (; runNext <= k; runNext++)
                {
                    int offset = (i * (1 + k) + runNext) * _vocab;
                    int nextId = Sample(seq, allLogits.Data.Slice(offset, _vocab));
                    if (TryEmitToken(seq, nextId)) { stopped = true; break; }
                    emitted++;

                    if (seq.AbandonToken.IsCancellationRequested)
                    {
                        Finish(seq, BatchFinishReason.Cancelled);
                        stopped = true;
                        break;
                    }

                    if (runNext < k)
                    {
                        if (nextId == seq.MtpDrafts![runNext])
                        {
                            seq.MtpAccepted++;
                            if (runNext == k - 1)
                            {
                                // The chain only predicted the last draft; feed it
                                // to the head now that the trunk kept it.
                                FeedLastMtpDraft(seq, trunkLen, k);
                                AdoptMtpHiddenRow(seq, k);
                            }
                            continue;
                        }

                        // Rejected: keep the verified prefix and forward the
                        // sampled token on the next step.
                        TruncateTrunkRow(seq, trunkLen + 1 + runNext);
                        _mtpCache!.TruncateRow(seq.Slot, mtpLen + 1 + runNext);
                        AdoptMtpHiddenRow(seq, runNext);
                        seq.PendingToken = nextId;
                        seq.State = SequenceState.Decoding;
                        break;
                    }

                    // All drafts held; this is the extra token after the run.
                    seq.PendingToken = nextId;
                    seq.State = SequenceState.Decoding;
                }

                if (stopped)
                {
                    int keep = Math.Min(runNext, k);
                    TruncateTrunkRow(seq, trunkLen + 1 + keep);
                    _mtpCache!.TruncateRow(seq.Slot, mtpLen + 1 + keep);
                    AdoptMtpHiddenRow(seq, keep);
                }
            }
        }

        Interlocked.Add(ref _decodedTokens, emitted);
        return true;
    }

    private void CaptureMtpHiddenAfterDecode(int count)
    {
        if (count == 1)
        {
            CaptureMtpHiddenFromLastNormed(_rowSeqs[0]);
            return;
        }

        for (int i = 0; i < count; i++)
            CaptureMtpHiddenRow(_rowSeqs[i], i);
    }

    private void FeedMtpPrefillChunk(BatchedSequence seq, Tensor<int> tokens, int position)
    {
        if (_mtpCache is null || _mtpView is null) { seq.MtpReady = false; return; }

        int n = _model.Config.HiddenDim;
        int len = tokens.Shape.Cols;
        using var hidden = _model.GetNormedHiddenStates();
        if (hidden.Shape[0] != 1 || hidden.Shape[1] != len || hidden.Shape[2] != n)
        {
            seq.MtpReady = false;
            return;
        }

        // The head input is the trunk hidden shifted one token left: the first
        // row pairs the chunk's first token with the previous chunk's last
        // hidden (zero for a fresh sequence).
        seq.MtpPrefillPrev ??= new float[n];
        using var mtpInput = new Tensor<float>(1, len, n);
        seq.MtpPrefillPrev.AsSpan(0, n).CopyTo(mtpInput.Data.Slice(0, n));
        for (int s = 1; s < len; s++)
            hidden.Data.Slice((s - 1) * n, n).CopyTo(mtpInput.Data.Slice(s * n, n));

        _mtpView.SetRows([seq.Slot]);
        _mtpView.SetPositions([position]);
        using (AcquireStepGate())
            _model.ForwardMtpFill(mtpInput, tokens, _mtpView, 0, null);

        hidden.Data.Slice((len - 1) * n, n).CopyTo(seq.MtpPrefillPrev);
        seq.MtpHidden ??= new float[n];
        hidden.Data.Slice((len - 1) * n, n).CopyTo(seq.MtpHidden);
        seq.MtpReady = true;
    }

    private void CaptureMtpHiddenFromLastNormed(BatchedSequence seq)
    {
        var normed = _model.LastNormed;
        if (normed is null) { seq.MtpReady = false; return; }
        int n = _model.Config.HiddenDim;
        seq.MtpHidden ??= new float[n];
        normed.Data[..n].CopyTo(seq.MtpHidden);
        seq.MtpReady = true;
    }

    private void CaptureMtpHiddenRow(BatchedSequence seq, int row)
    {
        int n = _model.Config.HiddenDim;
        seq.MtpHidden ??= new float[n];
        using var normed = _model.GetNormedHiddenRow(row);
        if (normed is null) { seq.MtpReady = false; return; }
        normed.Data[..n].CopyTo(seq.MtpHidden);
        seq.MtpReady = true;
    }

    private void CaptureMtpRunRows(BatchedSequence seq, int firstRow, int k)
    {
        int n = _model.Config.HiddenDim;
        int rows = k + 1;
        if (seq.MtpRunHidden is null || seq.MtpRunHidden.Length < rows * n)
            seq.MtpRunHidden = new float[rows * n];

        for (int r = 0; r < rows; r++)
        {
            using var normed = _model.GetNormedHiddenRow(firstRow + r);
            if (normed is null) { seq.MtpReady = false; return; }
            normed.Data[..n].CopyTo(seq.MtpRunHidden.AsSpan(r * n, n));
        }
    }

    private void AdoptMtpHiddenRow(BatchedSequence seq, int row)
    {
        int n = _model.Config.HiddenDim;
        if (seq.MtpRunHidden is null || row < 0 || (long)(row + 1) * n > seq.MtpRunHidden.Length)
        {
            seq.MtpReady = false;
            return;
        }

        seq.MtpHidden ??= new float[n];
        seq.MtpRunHidden.AsSpan(row * n, n).CopyTo(seq.MtpHidden);
        seq.MtpReady = true;
    }

    private void TruncateTrunkRow(BatchedSequence seq, int length)
    {
        for (int i = 0; i < _caches.Length; i++)
            _caches[i].TruncateRow(seq.Slot, length);
    }

    private void FeedLastMtpDraft(BatchedSequence seq, int trunkLen, int k)
    {
        if (_mtpCache is null || _mtpView is null || seq.MtpChainHidden is null || seq.MtpDrafts is null)
            return;

        int n = _model.Config.HiddenDim;
        using var hidden = new Tensor<float>(1, 1, n);
        seq.MtpChainHidden.AsSpan(0, n).CopyTo(hidden.Data);
        using var ids = new Tensor<int>(1, 1);
        ids.Data[0] = seq.MtpDrafts[k - 1];

        _mtpView.SetRows([seq.Slot]);
        _mtpView.SetPositions([trunkLen + k]);
        using (AcquireStepGate())
            _model.ForwardMtpHidden(hidden, ids, _mtpView, 0, null).Dispose();
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        float max = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > max) { max = values[i]; best = i; }
        return best;
    }

    private void SampleAndEmit(BatchedSequence seq, ReadOnlySpan<float> logitsRow)
    {
        int nextId = Sample(seq, logitsRow);
        if (TryEmitToken(seq, nextId)) return;

        int rowLength = _caches[0].LengthForRow(seq.Slot);
        if (rowLength >= Math.Min(_seqLen, _maxPosition) || seq.AbandonToken.IsCancellationRequested)
        {
            Finish(seq, seq.AbandonToken.IsCancellationRequested
                ? BatchFinishReason.Cancelled
                : BatchFinishReason.ContextFull);
            return;
        }

        seq.PendingToken = nextId;
        seq.State = SequenceState.Decoding;
    }

    /// <summary>
    /// Appends a sampled token to the sequence and streams its text. Returns true
    /// when the sequence ended at this token (stop token/string, max length,
    /// grammar exhaustion), so the caller must not use it as the next pending
    /// token.
    /// </summary>
    private bool TryEmitToken(BatchedSequence seq, int nextId)
    {
        if (nextId < 0) { Finish(seq, BatchFinishReason.GrammarExhausted); return true; }

        var gen = seq.Generation;
        seq.Generated.Add(nextId);

        string fragment = IncrementalDetokenizer.Next(
            _tokenizer, CollectionsMarshal.AsSpan(seq.Generated), ref seq.DetokPrefix, ref seq.DetokRead);
        seq.Decoded.Append(fragment);

        bool hitStop = false;
        int stopAt = int.MaxValue;
        int maxStopLen = seq.MaxStopLength;
        if (maxStopLen > 0 && seq.Decoded.Length >= maxStopLen)
        {
            if (seq.StopScratch is null || seq.StopScratch.Length < maxStopLen)
                seq.StopScratch = new char[maxStopLen];

            int start = seq.Decoded.Length - maxStopLen;
            seq.Decoded.CopyTo(start, seq.StopScratch, maxStopLen);
            ReadOnlySpan<char> tail = seq.StopScratch;
            foreach (string stop in gen.StopStrings)
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
            // Keep the text before the stop marker; it may fall mid-fragment.
            int fragStart = seq.Decoded.Length - fragment.Length;
            int keep = Math.Clamp(stopAt - fragStart, 0, fragment.Length);
            fragment = keep < fragment.Length ? fragment[..keep] : fragment;
            seq.Decoded.Length = fragStart + keep;
        }

        if (seq.Stream && fragment.Length > 0)
            seq.Emit(fragment);

        if (gen.StopTokenIds.Contains(nextId))
        {
            seq.EndedOnStop = true;
            Finish(seq, BatchFinishReason.StopToken);
            return true;
        }
        if (hitStop)
        {
            seq.EndedOnStop = true;
            Finish(seq, BatchFinishReason.StopString);
            return true;
        }
        if (seq.Generated.Count >= gen.MaxNewTokens)
        {
            Finish(seq, BatchFinishReason.MaxTokens);
            return true;
        }

        return false;
    }

    /// <summary>Samples one token, applying the repetition penalty and grammar exactly as the streaming generator does.</summary>
    private static int Sample(BatchedSequence seq, ReadOnlySpan<float> logitsRow)
    {
        var gen = seq.Generation;
        if (gen.RepetitionPenalty == 1.0f && seq.Sampling.Constraint is null)
            return Sampler.Sample(logitsRow, seq.Sampling, seq.Rng);

        int vocab = logitsRow.Length;
        if (seq.LogitScratch is null || seq.LogitScratch.Length < vocab)
            seq.LogitScratch = new float[vocab];

        Span<float> logits = seq.LogitScratch.AsSpan(0, vocab);
        logitsRow.CopyTo(logits);

        if (gen.RepetitionPenalty != 1.0f)
            seq.Penalty.Apply(logits, seq.PromptTokens, seq.Generated, gen.RepetitionPenalty, gen.RepetitionWindow);

        return SamplingHelpers.TrySampleConstrained(logits, seq.Sampling, seq.Rng, out int token) ? token : -1;
    }

    private void Finish(BatchedSequence seq, BatchFinishReason reason)
    {
        if (seq.State == SequenceState.Finished || seq.IsCompleted) return;
        seq.State = SequenceState.Finished;

        if (seq.Stream)
        {
            if (!seq.EndedOnStop && reason is not (BatchFinishReason.Cancelled or BatchFinishReason.Error))
            {
                string tail = IncrementalDetokenizer.Next(
                    _tokenizer, CollectionsMarshal.AsSpan(seq.Generated), ref seq.DetokPrefix, ref seq.DetokRead, final: true);
                if (tail.Length > 0)
                {
                    seq.Decoded.Append(tail);
                    seq.Emit(tail);
                }
            }
            seq.Complete(reason, seq.Decoded.ToString());
            return;
        }

        string text = reason == BatchFinishReason.Cancelled
            ? ""
            : _tokenizer.Decode(CollectionsMarshal.AsSpan(seq.Generated), skipSpecials: true).Replace("\uFFFD", "");
        if (text.Length > 0) seq.Emit(text);
        seq.Complete(reason, text);
    }

    private void CancelRemaining(BatchedSequence seq)
    {
        if (seq.IsCompleted) return;
        if (_fatal is not null) seq.Fail(_fatal);
        else Finish(seq, BatchFinishReason.Shutdown);
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(BatchedGenerator));

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => DisposeAsync().AsTask().GetAwaiter().GetResult();

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;
        try { _shutdown.Cancel(); } catch (ObjectDisposedException) { }
        _inbox.Writer.TryComplete();

        try { await _loop.ConfigureAwait(false); }
        catch (OperationCanceledException) { }

        for (int i = 0; i < _caches.Length; i++) _caches[i].Dispose();
        _mtpCache?.Dispose();
        _workspace.Dispose();
        _shutdown.Dispose();
    }
}
