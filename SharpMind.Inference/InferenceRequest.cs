namespace SharpMind.Inference;

// ─────────────────────────────────────────────────────────────────────────────
// ContinuousBatchScheduler
// Manages multiple concurrent requests sharing a single model forward pass.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// A single in-flight generation request managed by <see cref="ContinuousBatchScheduler"/>.
/// </summary>
public sealed class InferenceRequest
{
    private readonly TaskCompletionSource<string> _completion = new();
    private readonly List<int>                    _generatedIds = [];
    private readonly CancellationToken            _cancellationToken;

    internal InferenceRequest(
        int[]             promptIds,
        SamplingConfig    sampling,
        GenerationConfig  generation,
        CancellationToken cancellationToken)
    {
        PromptIds          = promptIds;
        Sampling           = sampling;
        Generation         = generation;
        _cancellationToken = cancellationToken;
        PositionOffset     = 0;
    }

    public int[]            PromptIds      { get; }
    public SamplingConfig   Sampling       { get; }
    public GenerationConfig Generation     { get; }
    public int              PositionOffset { get; internal set; }
    public bool             IsComplete     { get; private set; }
    public int              StepCount      => _generatedIds.Count;

    /// <summary>Awaitable result — resolves when generation completes.</summary>
    public Task<string> Result => _completion.Task;

    internal IReadOnlyList<int> GeneratedIds => _generatedIds;

    internal void AppendToken(int tokenId)
    {
        _generatedIds.Add(tokenId);

        if (Generation.StopTokenIds.Contains(tokenId) ||
            _generatedIds.Count >= Generation.MaxNewTokens)
            Complete();
    }

    internal void Complete()
    {
        if (IsComplete) return;
        IsComplete = true;
        _completion.TrySetResult(string.Empty); // decoded by caller
    }

    internal void Fail(Exception ex) => _completion.TrySetException(ex);

    internal bool IsCancelled => _cancellationToken.IsCancellationRequested;
}
