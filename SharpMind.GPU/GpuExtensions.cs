using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
namespace SharpMind.GPU;

public static class MappingBuilderExtensions
{
    private static readonly string[] HwSuffixes = ["avx2", "fma", "sse", "scalar"];

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
    private static readonly string[] VecDotKeys = [
        QuantizationKeys.KeyVecDotQ3K,
        QuantizationKeys.KeyVecDotQ4K,
        QuantizationKeys.KeyVecDotQ5K,
        QuantizationKeys.KeyVecDotQ6K,
        QuantizationKeys.KeyVecDotQ8_0,
        QuantizationKeys.KeyVecDotQ4_0,
        QuantizationKeys.KeyVecDotQ4_1,
        QuantizationKeys.KeyVecDotQ5_0,
        QuantizationKeys.KeyVecDotQ5_1,
        QuantizationKeys.KeyVecDotQ8_1,
        QuantizationKeys.KeyVecDotQ2K,
        QuantizationKeys.KeyVecDotQ8K,
    ];

    private static readonly string[] QmmKeys = [
        QuantizationKeys.KeyQuantizedMatMulQ8_0,
        QuantizationKeys.KeyQuantizedMatMulQ5_0,
        QuantizationKeys.KeyQuantizedMatMulQ6K,
        QuantizationKeys.KeyQuantizedMatMulQ4_0,
        QuantizationKeys.KeyQuantizedMatMulQ4_1,
        QuantizationKeys.KeyQuantizedMatMulQ2K,
        QuantizationKeys.KeyQuantizedMatMulQ3K,
        QuantizationKeys.KeyQuantizedMatMulQ4K,
        QuantizationKeys.KeyQuantizedMatMulQ5K,
        QuantizationKeys.KeyQuantizedMatMulQ8K,
        QuantizationKeys.KeyQuantizedMatMulQ8_1,
        QuantizationKeys.KeyQuantizedMatMulQ5_1,
    ];

    public static MappingBuilder WithGpu(this MappingBuilder builder)
    {
        TryOverrideGpu(builder, SharpMindConfig.KeyPointWise);
        TryOverrideGpu(builder, SharpMindConfig.KeyGate);

        builder.Override(SharpMindConfig.KeySoftmax, GPUSharpMindConfig.ValSoftmax);
        builder.Override(SharpMindConfig.KeyRMSNorm, GPUSharpMindConfig.ValRMSNorm);

        foreach (var key in VecDotKeys)
            TryOverrideGpu(builder, key);

        foreach (var key in QmmKeys)
            TryOverrideGpu(builder, key);

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
