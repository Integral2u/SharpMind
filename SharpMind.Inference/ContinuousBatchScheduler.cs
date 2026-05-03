namespace SharpMind.Inference;

/// <summary>
/// Continuous batching scheduler — processes multiple requests in parallel
/// by batching their current decode tokens into a single forward pass.
///
/// Unlike static batching (where all sequences must have the same length),
/// continuous batching allows new requests to join and completed requests to
/// leave mid-generation. Each request has its own KV-cache slice.
///
/// This is the approach used by vLLM, TGI, and production inference servers.
///
/// Usage:
/// <code>
/// var scheduler = new ContinuousBatchScheduler(model, tokenizer, ops,
///                     maxConcurrent: 8);
/// await scheduler.StartAsync();
///
/// // Submit requests from multiple threads/tasks
/// var result = await scheduler.SubmitAsync("Tell me a story");
/// </code>
/// </summary>
public sealed class ContinuousBatchScheduler : IAsyncDisposable
{
    private readonly Model.Transformer _model;
    private readonly Tokenizer.Tokenizer _tokenizer;
    private readonly InferenceOps                _ops;
    private readonly int                         _maxConcurrent;
    private readonly System.Threading.Channels.Channel<InferenceRequest>   _requestChannel;
    private CancellationTokenSource?             _cts;
    private Task?                                _runLoop;
    private bool                                 _disposed;

    // Per-request KV-cache slices — pre-allocated pool
    private readonly KvCache[] _cachePool;

    public ContinuousBatchScheduler(
        SharpMind.Model.Transformer       model,
        SharpMind.Tokenizer.Tokenizer     tokenizer,
        InferenceOps                      ops,
        int                               maxConcurrent = 8)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(ops);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxConcurrent);

        _model          = model;
        _tokenizer      = tokenizer;
        _ops            = ops;
        _maxConcurrent  = maxConcurrent;
        _requestChannel = System.Threading.Channels.Channel.CreateBounded<InferenceRequest>(
            new System.Threading.Channels.BoundedChannelOptions(maxConcurrent * 4)
            {
                FullMode    = System.Threading.Channels.BoundedChannelFullMode.Wait,
                SingleReader = true,
            });

        _cachePool = [.. Enumerable.Range(0, maxConcurrent).Select(_ => new KvCache(model.Config))];
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// <summary>Starts the background scheduling loop.</summary>
    public Task StartAsync(CancellationToken cancellationToken = default)
    {
        _cts     = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _runLoop = Task.Run(() => RunLoopAsync(_cts.Token), _cts.Token);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Submits a generation request and returns a task that resolves when complete.
    /// </summary>
    public async Task<string> SubmitAsync(
        string            prompt,
        SamplingConfig?   sampling   = null,
        GenerationConfig? generation = null,
        CancellationToken cancellationToken = default)
    {
        if (_runLoop is null)
            throw new InvalidOperationException(
                "Call StartAsync before submitting requests.");

        int[] promptIds = _tokenizer.Encode(prompt, addBos: true, addEos: false);
        var   request   = new InferenceRequest(
            promptIds,
            sampling   ?? SamplingConfig.Greedy,
            generation ?? GenerationConfig.Default,
            cancellationToken);

        await _requestChannel.Writer.WriteAsync(request, cancellationToken);
        string result = await request.Result;

        return _tokenizer.Decode([.. request.GeneratedIds], skipSpecials: true);
    }

    // ── Background loop ───────────────────────────────────────────────────

    private async Task RunLoopAsync(CancellationToken cancellationToken)
    {
        var active     = new List<(InferenceRequest Request, int CacheIdx)>(_maxConcurrent);
        var freeCache  = new Stack<int>(Enumerable.Range(0, _maxConcurrent));

        while (!cancellationToken.IsCancellationRequested)
        {
            // Fill batch with waiting requests up to maxConcurrent
            while (active.Count < _maxConcurrent &&
                   _requestChannel.Reader.TryRead(out var newReq))
            {
                if (freeCache.Count == 0) break;
                int cacheIdx = freeCache.Pop();
                _cachePool[cacheIdx].Reset();
                active.Add((newReq, cacheIdx));
            }

            if (active.Count == 0)
            {
                // No active requests — wait for one
                try
                {
                    var next = await _requestChannel.Reader.ReadAsync(cancellationToken);
                    if (freeCache.Count > 0)
                    {
                        int idx = freeCache.Pop();
                        _cachePool[idx].Reset();
                        active.Add((next, idx));
                    }
                }
                catch (OperationCanceledException) { break; }
                continue;
            }

            // One decode step for all active requests
            for (int i = active.Count - 1; i >= 0; i--)
            {
                var (req, cacheIdx) = active[i];

                if (req.IsCancelled)
                {
                    req.Complete();
                    freeCache.Push(cacheIdx);
                    active.RemoveAt(i);
                    continue;
                }

                // Get the current token (last generated, or last prompt token for first step)
                int currentToken = req.StepCount == 0
                    ? req.PromptIds[^1]
                    : req.GeneratedIds[^1];

                // Single-token forward — use decode attention path
                using var input   = SharpMind.Core.Tensors.Tensor<int>.From([currentToken], 1, 1);
                int        pos    = req.PositionOffset + req.PromptIds.Length + req.StepCount - 1;
                using var logits  = _model.Forward(input, pos);

                int vocabSize = logits.Shape[2];
                var logitSpan = logits.Data[..vocabSize];

                int nextToken = Sampler.Sample(
                    System.Runtime.InteropServices.MemoryMarshal.Cast<float, float>(logitSpan),
                    req.Sampling);

                req.AppendToken(nextToken);

                if (req.IsComplete)
                {
                    freeCache.Push(cacheIdx);
                    active.RemoveAt(i);
                }
            }

            // Yield to allow new requests to be submitted
            await Task.Yield();
        }

        // Complete any remaining requests on shutdown
        foreach (var (req, _) in active)
            req.Complete();
    }

    // ── Disposal ──────────────────────────────────────────────────────────

    public async ValueTask DisposeAsync()
    {
        if (_disposed) return;
        _disposed = true;

        _cts?.Cancel();
        _requestChannel.Writer.TryComplete();

        if (_runLoop is not null)
            await _runLoop.ConfigureAwait(false);

        foreach (var cache in _cachePool) cache.Dispose();
        _cts?.Dispose();
    }

}
