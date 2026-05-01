using JigSawDotNet;

namespace SharpMind.Training;

/// <summary>
/// Assembles a <see cref="TrainingOps"/> instance from a <see cref="SharpMindConfig"/>.
/// Hardware tier is resolved once here — no runtime checks inside any kernel.
///
/// Usage:
/// <code>
/// var trainingOps = TrainingOpsFactory.Create(SharpMindConfig.Llama);
/// var optimizer   = new AdamW(parameters, trainingOps, lr: 3e-4f);
/// </code>
/// </summary>
public static class TrainingOpsFactory
{
    public static TrainingOps Create(SharpMindConfig config)
        => Assembler.CreateInstance<TrainingOps>(config.ToJigSawMapping());
}
