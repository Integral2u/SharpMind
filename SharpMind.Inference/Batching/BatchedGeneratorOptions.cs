namespace SharpMind.Inference;

/// <summary>
/// How a batched sequence ended. Mirrors the stops <see cref="StandardGenerator{T}"/>
/// honours, so a host can map the same reasons regardless of which generator ran.
/// </summary>
public enum BatchFinishReason
{
    /// <summary>Still running (or never started).</summary>
    None = 0,
    /// <summary>The model produced a token listed in <see cref="GenerationConfig.StopTokenIds"/>.</summary>
    StopToken,
    /// <summary>A <see cref="GenerationConfig.StopStrings"/> entry appeared in the decoded text.</summary>
    StopString,
    /// <summary><see cref="GenerationConfig.MaxNewTokens"/> tokens were generated.</summary>
    MaxTokens,
    /// <summary>The model's context window (or the cache) ran out.</summary>
    ContextFull,
    /// <summary>The caller cancelled or abandoned the sequence.</summary>
    Cancelled,
    /// <summary>Generation failed; see <see cref="BatchedSequence.Error"/>.</summary>
    Error,
    /// <summary>A grammar constraint had no legal continuation left.</summary>
    GrammarExhausted,
    /// <summary>The generator was disposed while the sequence was in flight.</summary>
    Shutdown,
}

/// <summary>
/// Configuration for <see cref="BatchedGenerator"/>. Batches trade latency for
/// throughput: <see cref="MaxBatch"/> sequences share one forward pass per
/// decode step, so the cost of a step is roughly the cost of one sequence's step
/// until the kernels become compute-bound.
/// </summary>
public sealed record BatchedGeneratorOptions
{
    /// <summary>Sequences decoded together in one forward pass. Default 8.</summary>
    public int MaxBatch { get; init; } = 8;

    /// <summary>
    /// Cache capacity per row, in tokens. Null uses
    /// <see cref="ModelConfig.ComputeMaxCacheLength"/> for the model.
    /// </summary>
    public int? MaxSeqLen { get; init; }

    /// <summary>
    /// Prompt tokens fed per prefill forward. The workspace is sized for
    /// <see cref="Prefill.MaxChunkLength"/>; a larger value needs a larger
    /// workspace, so the default is the same bound.
    /// </summary>
    public int MaxChunkLength { get; init; } = Prefill.MaxChunkLength;

    /// <summary>
    /// Requests allowed to wait for a free row. Beyond this
    /// <see cref="EnqueueTokensAsync"/> throws, rather than letting an unbounded
    /// queue turn overload into timeouts.
    /// </summary>
    public int MaxQueued { get; init; } = 64;

    /// <summary>Prepend BOS when enqueuing text (token-id callers control their own).</summary>
    public bool AddBos { get; init; }

    /// <summary>Append EOS when enqueuing text.</summary>
    public bool AddEos { get; init; }

    /// <summary>Seed for sampling when a request does not supply one.</summary>
    public int? Seed { get; init; }
}

/// <summary>Aggregate throughput counters for a running batcher.</summary>
/// <param name="Steps">Decode steps completed (one forward pass each).</param>
/// <param name="GeneratedTokens">Tokens produced across every sequence.</param>
/// <param name="DecodeSeconds">Wall time spent inside decode steps.</param>
/// <param name="TokensPerSecond">Aggregate decode tokens per second, not per sequence.</param>
/// <param name="AverageBatchSize">Mean sequences per decode step.</param>
public readonly record struct BatchedGeneratorStats(
    long Steps,
    long GeneratedTokens,
    double DecodeSeconds,
    double TokensPerSecond,
    double AverageBatchSize);

/// <summary>
/// Raised when a batch has no free row and the wait queue is at its limit. The
/// caller should refuse the request (HTTP 429) instead of parking it.
/// </summary>
public sealed class BatchQueueFullException(int maxBatch, int maxQueued)
    : Exception($"Batch is at capacity ({maxBatch} active, {maxQueued} queued); retry shortly.")
{
    public int MaxBatch { get; } = maxBatch;
    public int MaxQueued { get; } = maxQueued;
}
