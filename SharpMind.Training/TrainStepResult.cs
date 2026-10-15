namespace SharpMind.Training;

/// <summary>
/// Step-level event data passed to the progress callback.
/// </summary>
public sealed record TrainStepResult
{
    public int   Step            { get; init; }
    public float Loss            { get; init; }
    public float LearningRate    { get; init; }
    public float GradNorm        { get; init; }
    public TimeSpan StepTime     { get; init; }
    public int   TokensThisStep  { get; init; }
    public long  TokensSeen      { get; init; }
    public float TokensPerSecond { get; init; }
    public float? ValLoss        { get; init; }
}
