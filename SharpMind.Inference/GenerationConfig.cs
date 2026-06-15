namespace SharpMind.Inference;

/// <summary>
/// Controls how token generation stops and any post-processing applied
/// during the generation loop.
/// </summary>
public sealed record GenerationConfig
{
    // Stopping

    /// <summary>Maximum number of new tokens to generate.</summary>
    public int MaxNewTokens { get; init; } = 256;

    /// <summary>
    /// Token IDs that stop generation when produced.
    /// Typically contains EosId. Can include model-specific stop tokens
    /// like &lt;|eot_id|&gt; for LLaMA 3 instruct.
    /// </summary>
    public IReadOnlyList<int> StopTokenIds { get; init; } = [];

    /// <summary>
    /// Stop when any of these strings appears in the decoded output.
    /// Useful for chat templates that end with "\n\nHuman:" etc.
    /// </summary>
    public IReadOnlyList<string> StopStrings { get; init; } = [];

    // Repetition penalty

    /// <summary>
    /// Penalises tokens that have already appeared in the context.
    /// 1.0 = no penalty. Values > 1 reduce repetition.
    /// Applied multiplicatively to the logit before sampling.
    /// Typical: 1.1–1.3.
    /// </summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>
    /// Number of most recent tokens to consider for repetition penalty.
    /// 0 = apply penalty over the full context window.
    /// </summary>
    public int RepetitionWindow { get; init; } = 0;

    /// <summary>
    /// When the KV-cache fills, trim to keep this many recent tokens.
    /// 0 = keep half the cache (default sliding window behaviour).
    /// </summary>
    public int SlidingWindowSize { get; init; } = 0;
    // Streaming

    /// <summary>
    /// When true, each token is decoded and yielded as a partial string.
    /// When false, generation completes before any output is returned.
    /// </summary>
    public bool Stream { get; init; } = true;

    // Presets

    public static GenerationConfig Default => new();

    public static GenerationConfig Chat(int eosId) => new()
    {
        MaxNewTokens     = 1024,
        StopTokenIds     = [eosId],
        RepetitionPenalty = 1.1f,
        Stream           = true,
    };

    public static GenerationConfig Completion => new()
    {
        MaxNewTokens = 512,
        Stream       = true,
    };
}
