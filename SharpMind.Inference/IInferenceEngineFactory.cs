using SharpMind.Core;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <param name="Model">The model to run inference for.</param>
/// <param name="Config">
/// Architecture/kernel configuration the model was built with. Needed separately from
/// <paramref name="Model"/> for the same reason <c>TrainingEngineContext.Config</c> is:
/// e.g. a <c>GatedFfnLayer</c> resolves its gate activation (SiLU/GeGLU) into an opaque
/// <c>ActivationOps</c> at load time and does not expose which kind it picked, so a
/// device engine that needs to pick its own matching kernel has to be told separately.
/// </param>
/// <param name="MaxCacheLength">Cache capacity in positions (<c>ModelConfig.ComputeMaxCacheLength</c>).</param>
public sealed record InferenceEngineContext(Transformer Model, SharpMindConfig Config, int MaxCacheLength);

/// <summary>
/// Inference capability of an accelerator plugin (<c>IAcceleratorPlugin.Capabilities</c>).
/// Same refusal contract as <c>ITrainingEngineFactory</c>: null + a reason the caller
/// surfaces to the user, never a silent CPU fallback.
/// </summary>
public interface IInferenceEngineFactory
{
    IInferenceEngine? TryCreate(InferenceEngineContext context, out string? reason);
}
