namespace SharpMind.Training;

/// <summary>
/// Validation/evaluation result for a checkpoint step.
/// </summary>
public sealed record EvalResult
{
    public float Loss { get; init; } = float.NaN;
    public float Perplexity { get; init; } = float.NaN;
}
