using SharpMind.Core;

namespace SharpMind.Training;

/// <summary>
/// Options for loading an .SMM file into a trainable model
/// (see <see cref="SmmTrainingPipeline.LoadForTraining"/>).
/// </summary>
public sealed record TrainingLoadOptions
{
    /// <summary>
    /// Hardware tier for the derived <see cref="SharpMindConfig"/>. The default
    /// (<see cref="HardwareTier.Auto"/>) resolves to the scalar backend, which is
    /// what the training forward/backward path runs on.
    /// </summary>
    public HardwareTier Hardware { get; init; } = HardwareTier.Auto;

    /// <summary>Optional progress sink for the weight-initialisation load.</summary>
    public IProgress<float>? Progress { get; init; }
}
