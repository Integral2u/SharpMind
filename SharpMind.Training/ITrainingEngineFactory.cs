using SharpMind.Core;
using SharpMind.Core.Training;
using SharpMind.Model;

namespace SharpMind.Training;

/// <summary>
/// Everything an engine needs to attach to a training run. A record rather than
/// a parameter list so third-party plugins keep compiling when a field is added.
/// </summary>
/// <param name="Model">The model being trained.</param>
/// <param name="Parameters">Trainable parameters, in <c>Transformer.Parameters()</c> order.</param>
/// <param name="Mapping">The gradient kernel mapping resolved for <paramref name="Config"/>.</param>
/// <param name="Config">Architecture/kernel configuration.</param>
/// <param name="Loss">The loss the CPU path would use; device engines may fuse their own equivalent.</param>
/// <param name="BatchSize">Batch rows the run will present.</param>
/// <param name="SeqLen">Sequence length the run will present.</param>
/// <param name="IgnoreId">Label id excluded from the loss.</param>
/// <param name="LabelSmoothing">Label smoothing the run was configured with.</param>
/// <remarks>
/// <paramref name="IgnoreId"/> and <paramref name="LabelSmoothing"/> must agree with
/// whatever <paramref name="Loss"/> was itself constructed with — they are carried
/// here separately only because <see cref="ILoss{TLabel}"/> does not expose them
/// for a device engine to read back.
/// </remarks>
public sealed record TrainingEngineContext(
    Transformer              Model,
    IReadOnlyList<Parameter> Parameters,
    GradientMapping          Mapping,
    SharpMindConfig          Config,
    ILoss<int>               Loss,
    int                      BatchSize,
    int                      SeqLen,
    int                      IgnoreId       = -100,
    float                    LabelSmoothing = 0f);

/// <summary>
/// Training capability of an accelerator plugin (see
/// <c>SharpMind.Core.Plugins.IAcceleratorPlugin.Capabilities</c>).
/// </summary>
public interface ITrainingEngineFactory
{
    /// <summary>
    /// Creates an engine for <paramref name="context"/>, or returns null with a
    /// human-readable <paramref name="reason"/> when this accelerator cannot train
    /// that model/config (unsupported layer kind, no device, too large…). The
    /// caller owns the returned engine.
    /// </summary>
    ITrainingEngine? TryCreate(TrainingEngineContext context, out string? reason);
}
