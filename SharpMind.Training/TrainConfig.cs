namespace SharpMind.Training;

/// <summary>
/// Configuration for a training run.
/// </summary>
public sealed record TrainConfig
{
    /// <summary>Total number of gradient update steps.</summary>
    public int TotalSteps { get; init; } = 10_000;

    /// <summary>
    /// Number of batches to accumulate gradients over before one optimizer step.
    /// Effective batch size = DataLoader batch size × GradAccumSteps.
    /// </summary>
    public int GradAccumSteps { get; init; } = 1;

    /// <summary>Maximum global gradient L2 norm. 0 = no clipping.</summary>
    public float GradClipNorm { get; init; } = 1.0f;

    /// <summary>Log loss every N steps.</summary>
    public int LogInterval { get; init; } = 100;

    /// <summary>Save a checkpoint every N steps. 0 = never.</summary>
    public int CheckpointInterval { get; init; } = 1_000;

    /// <summary>Directory to write checkpoints.</summary>
    public string CheckpointDir { get; init; } = "checkpoints";

    /// <summary>Resume from this checkpoint directory. Null = train from scratch.</summary>
    public string? ResumeFrom { get; init; }
}
