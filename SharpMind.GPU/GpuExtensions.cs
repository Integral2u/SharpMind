using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;

namespace SharpMind.GPU;

public static class MappingBuilderExtensions
{
    private static readonly string[] HwSuffixes = ["avx2", "fma", "sse"];

    /// <summary>
    /// Augments the mapping with GPU-accelerated kernels for activations,
    /// gates, softmax, and RMSNorm.
    ///
    /// Call after <see cref="MappingBuilder.ApplyPreset"/> so that the base
    /// CPU mapping values are already set. This method strips the CPU hardware
    /// suffix (avx2/fma/sse) and replaces it with "gpu".
    ///
    /// JigSaw discovers the GPU <c>[PuzzlePeice]</c> entries at assembly time
    /// via <c>AppDomain</c> scanning, so no special assembly reference is needed
    /// beyond whichever project calls <c>WithGpu()</c> (and thereby loads
    /// SharpMind.GPU into the process).
    /// </summary>
    public static MappingBuilder WithGpu(this MappingBuilder builder)
    {
        TryOverrideGpu(builder, SharpMindConfig.KeyPointWise);
        TryOverrideGpu(builder, SharpMindConfig.KeyGate);

        builder.Override(SharpMindConfig.KeySoftmax, GPUSharpMindConfig.ValSoftmax);
        builder.Override(SharpMindConfig.KeyRMSNorm, GPUSharpMindConfig.ValRMSNorm);

        return builder;
    }

    /// <summary>
    /// If <paramref name="key"/> has a value in the builder's mapping, strips
    /// any known CPU hardware suffix and appends <c>"gpu"</c>.
    /// </summary>
    private static void TryOverrideGpu(MappingBuilder builder, string key)
    {
        if (!builder.TryGetValue(key, out var val) || val is null)
            return;

        string stripped = val;
        foreach (var sfx in HwSuffixes)
        {
            if (stripped.EndsWith(sfx) && stripped.Length > sfx.Length)
            {
                stripped = stripped[..^sfx.Length];
                break;
            }
        }
        builder.Override(key, stripped + "gpu");
    }
}
