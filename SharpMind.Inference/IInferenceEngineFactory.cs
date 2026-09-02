using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

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

    /// <summary>
    /// Metadata-only, pre-weight-load compatibility check. The host calls this once it has read a
    /// model's headers (dtype usage, architecture config) but before materializing the weights, so
    /// a model this engine cannot run fails fast instead of loading the whole file then refusing.
    /// Returns null when this factory has no verdict (either it's not applicable, or the caller
    /// should still attempt <see cref="TryCreate"/> for the authoritative per-layer gate); a
    /// non-null string is a refusal reason the host surfaces to the user.
    /// </summary>
    string? CheckSupported(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config) => null;

    /// <summary>
    /// Metadata-only, pre-weight-load description of which model content this engine would run on the
    /// CPU instead of the device. Used when <see cref="CheckSupported"/> accepts the model but running
    /// it would mix device and host tensors (e.g. a GGUF whose block linears use a quant dtype that has
    /// no on-device kernel: the engine runs those tensors on the host and the rest on the device).
    /// Returns null when the engine can run the whole model on the device (no fallback, no consent
    /// needed); a non-null string is a human-readable description of what falls back, which the host
    /// surfaces once before loading weights if the user has not already consented.
    /// </summary>
    string? DescribeCpuFallback(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config) => null;
}
