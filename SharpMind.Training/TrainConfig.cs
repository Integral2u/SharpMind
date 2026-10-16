using SharpMind.Core.Quantization;

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

    /// <summary>
    /// Label smoothing epsilon for the default cross-entropy loss. 0 = disabled.
    /// Typical value 0.1. Applies only when no explicit loss is passed to the
    /// <see cref="TrainLoop"/> constructor.
    /// </summary>
    public float LabelSmoothing { get; init; } = 0f;

    /// <summary>Log loss every N steps.</summary>
    public int LogInterval { get; init; } = 100;

    /// <summary>Save a checkpoint every N steps. 0 = never.</summary>
    public int CheckpointInterval { get; init; } = 1_000;

    /// <summary>Directory to write checkpoints.</summary>
    public string CheckpointDir { get; init; } = "checkpoints";

    /// <summary>Resume from this checkpoint directory. Null = train from scratch.</summary>
    public string? ResumeFrom { get; init; }

    /// <summary>
    /// Apply the checkpoint's data cursor on resume: skip the batches the
    /// interrupted run already consumed, so the resumed run trains on what comes
    /// next instead of re-training the head of the corpus it has seen. Defaults
    /// to true - the correct behaviour whenever the resumed run replays the same
    /// data stream.
    ///
    /// Set false when the stream itself differs from the one the cursor was
    /// recorded against - an incremental run, which by construction feeds only
    /// new and changed files, so skipping would throw away the very data the run
    /// exists to train on.
    /// </summary>
    public bool ResumeDataCursor { get; init; } = true;

    /// <summary>
    /// Seed of the batcher's RNG, when the run declares one. Recorded in every
    /// checkpoint and checked on resume: a cursor computed under seed A cannot
    /// replay a stream produced by seed B, so a mismatch fails loudly instead of
    /// silently training on the wrong data. Null (default) skips both the record
    /// and the check.
    /// </summary>
    public int? DataSeed { get; init; }

    /// <summary>
    /// How many rolling (step-marked) checkpoints to retain. 0 = prune none
    /// (keep every interval checkpoint). N &gt; 0 = keep only the newest N
    /// step checkpoints, deleting the older ones as the run progresses. A
    /// negative value disables rolling checkpoints entirely — only the final
    /// checkpoint ("-final") is saved. The final checkpoint is never pruned.
    /// </summary>
    public int KeepRecent { get; init; }

    /// <summary>
    /// Quantization-aware training target. Null or <see cref="QuantDType.F32"/>
    /// disables QAT (pure float training). <see cref="QuantDType.F16"/> is always
    /// safe. <see cref="QuantDType.Q8_0"/> and <see cref="QuantDType.Q4_0"/>
    /// require every linear-layer weight dimension to be a multiple of 32 — the
    /// linear layers throw at training time if a block-quantized shape is invalid.
    /// Master weights stay F32; quantization error is injected in each forward pass
    /// and gradients flow straight through to the master weight.
    /// </summary>
    public QuantDType? QuantAwareTraining { get; init; }

    /// <summary>
    /// Evaluate the model every N steps. 0 = never. When greater than 0,
    /// <see cref="TrainLoop.RunAsync"/> will invoke the optional <c>evaluate</c>
    /// callback at the configured interval (after the optimizer step, before
    /// checkpoints) and propagate the resulting validation loss into the
    /// <see cref="TrainStepResult.ValLoss"/> for that logged step.
    /// </summary>
    public int EvalInterval { get; init; }
}
