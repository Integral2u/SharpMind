namespace SharpMind.Inference;

/// <summary>
/// Controls how the next token is selected from the logits distribution.
/// All strategies are pure functions of the logit vector — no state.
/// </summary>
public sealed record SamplingConfig
{
    // Strategy selection

    /// <summary>
    /// Temperature to apply before sampling. Values:
    ///   0   → greedy (argmax, no randomness)
    ///   1   → sample from the unmodified distribution
    ///  >1   → flatter distribution (more random)
    ///  &lt;1  → sharper distribution (more peaked)
    /// </summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>
    /// Top-k filtering: keep only the k highest-probability tokens before sampling.
    /// 0 = disabled. Typical values: 40–100.
    /// </summary>
    public int TopK { get; init; } = 0;

    /// <summary>
    /// Top-p (nucleus) sampling: keep the smallest set of tokens whose cumulative
    /// probability exceeds p. 1.0 = disabled. Typical: 0.9–0.95.
    /// </summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>
    /// Min-p filtering: discard tokens whose probability is less than
    /// min_p × max_probability. Simpler and often better than top-p.
    /// 0 = disabled. Typical: 0.05–0.1.
    /// </summary>
    public float MinP { get; init; } = 0.0f;

    /// <summary>Seed for reproducible sampling. Null = non-deterministic.</summary>
    public int? Seed { get; init; }

    // Presets

    /// <summary>Deterministic greedy decoding — always picks the highest-probability token.</summary>
    public static SamplingConfig Greedy => new() { Temperature = 0f };

    /// <summary>LLaMA 3 chat defaults: temperature 0.6, top-p 0.9.</summary>
    public static SamplingConfig Llama3Chat => new() { Temperature = 0.6f, TopP = 0.9f };

    /// <summary>Creative writing: higher temperature, broad nucleus.</summary>
    public static SamplingConfig Creative => new() { Temperature = 0.9f, TopP = 0.95f, MinP = 0.05f };
}
