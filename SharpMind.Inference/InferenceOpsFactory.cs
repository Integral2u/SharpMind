using JigSawDotNet;

namespace SharpMind.Inference;

/// <summary>
/// Assembles an <see cref="InferenceOps"/> from a <see cref="SharpMindConfig"/>
/// and an <see cref="InferenceConfig"/>.
/// The full mapping merges base hardware keys with inference-specific keys
/// so one assembled type covers attention algorithm, quantization, and hw tier.
/// </summary>
public static class InferenceOpsFactory
{
    public static InferenceOps Create(SharpMindConfig sharpConfig, InferenceConfig inferConfig)
    {
        var mapping = sharpConfig.ToJigSawMapping();

        // Merge inference-specific keys — overrides any conflicting base keys
        foreach (var kv in inferConfig.ToJigSawMapping(sharpConfig.ResolvedHardware))
            mapping[kv.Key] = kv.Value;

        return Assembler.CreateInstance<InferenceOps>(mapping);
    }
}
