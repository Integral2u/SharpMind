namespace SharpMind.Training;

/// <summary>
/// Outcome of a <see cref="FiniteDifferenceTrainer.TrainAsync"/> run.
/// </summary>
public sealed record FiniteDifferenceResult
{
    /// <summary>Loss at the final step (NaN if no steps ran).</summary>
    public float FinalLoss { get; init; } = float.NaN;

    /// <summary>Number of optimizer steps performed.</summary>
    public int Steps { get; init; }

    /// <summary>Total wall-clock time spent training.</summary>
    public TimeSpan Elapsed { get; init; }

    /// <summary>Last checkpoint directory written, if any.</summary>
    public string? CheckpointPath { get; init; }
}
