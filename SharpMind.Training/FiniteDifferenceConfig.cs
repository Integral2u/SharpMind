namespace SharpMind.Training;

/// <summary>
/// Configuration for a <see cref="FiniteDifferenceTrainer"/> run.
/// </summary>
public sealed record FiniteDifferenceConfig
{
    /// <summary>Total number of gradient update steps.</summary>
    public int TotalSteps { get; init; } = 100;

    /// <summary>Invoke the step callback every N steps.</summary>
    public int LogInterval { get; init; } = 20;

    /// <summary>Perturbation step h used for the central-difference gradient estimate.</summary>
    public float Perturbation { get; init; } = 1e-3f;

    /// <summary>Save a checkpoint every N steps. 0 = never.</summary>
    public int CheckpointInterval { get; init; }

    /// <summary>Directory to write checkpoints.</summary>
    public string CheckpointDir { get; init; } = "checkpoints";
}
