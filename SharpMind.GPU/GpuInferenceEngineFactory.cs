using SharpMind.Inference;

namespace SharpMind.GPU;

/// <summary>
/// Inference capability of <see cref="IlgpuAcceleratorPlugin"/>. Same device-acquisition and
/// exception-to-reason conversions as <see cref="GpuTrainingEngineFactory"/> — see its doc
/// comment. M0 hybrid: <see cref="GpuInferenceEngine"/> accelerates the first prefill on the
/// device and runs continued prefill/decode on the CPU; <see cref="MaxPromptTokens"/> bounds
/// the GPU prefill's arena because the engine does not chunk long GPU prompts yet.
/// </summary>
public sealed class GpuInferenceEngineFactory : IInferenceEngineFactory
{
    /// <summary>Upper bound on a single GPU Prefill call's token count. See <see cref="GpuInferenceEngine"/>'s constructor doc.</summary>
    public int MaxPromptTokens { get; init; } = 4096;

    public IInferenceEngine? TryCreate(InferenceEngineContext context, out string? reason)
    {
        ArgumentNullException.ThrowIfNull(context);

        GpuDevice? device = GpuDevice.TryCreate(out reason);
        if (device is null) return null;

        try
        {
            var engine = new GpuInferenceEngine(
                device,
                context.Model,
                context.Config,
                context.MaxCacheLength,
                Math.Min(MaxPromptTokens, context.MaxCacheLength));

            reason = null;
            device = null;          // ownership handed to the engine
            return engine;
        }
        catch (NotSupportedException ex)
        {
            // The engine names the exact shape it refuses (LayerNorm, MoE, dense FFN,
            // quantization, …), same as GpuTrainingEngineFactory.
            reason = ex.Message;
            return null;
        }
        finally
        {
            device?.Dispose();      // only runs when we did not hand it over
        }
    }
}
