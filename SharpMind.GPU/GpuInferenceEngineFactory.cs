using SharpMind.Core;
using SharpMind.Inference;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.GPU;

/// <summary>
/// Inference capability of <see cref="IlgpuAcceleratorPlugin"/>. Same device-acquisition and
/// exception-to-reason conversions as <see cref="GpuTrainingEngineFactory"/> — see its doc
/// comment. <see cref="GpuInferenceEngine"/> runs the whole forward on the device: first prefill,
/// continued prefill, and KV-cache-efficient decode. <see cref="MaxPromptTokens"/> bounds the
/// arena for one GPU prefill call's new tokens, because the engine does not chunk long GPU prompts
/// yet — a call that would overflow it, or overrun the device cache, falls back to the CPU
/// <see cref="SharpMind.Model.Transformer"/>.
/// </summary>
public sealed class GpuInferenceEngineFactory : IInferenceEngineFactory
{
    /// <summary>Upper bound on a single GPU Prefill call's token count. See <see cref="GpuInferenceEngine"/>'s constructor doc.</summary>
    public int MaxPromptTokens { get; init; } = 4096;

    public string? CheckSupported(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config)
        => GpuInferenceEngine.CheckSupported(meta, modelConfig, config, out var reason) ? null : reason;

    public string? DescribeCpuFallback(ModelMetaData meta, ModelConfig modelConfig, SharpMindConfig config)
        => GpuInferenceEngine.DescribeCpuFallback(meta, modelConfig, config);

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
