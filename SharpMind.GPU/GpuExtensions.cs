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
    private static readonly string[] VecDotKeys = [
        QuantizationConfig.KeyVecDotQ3K,
        QuantizationConfig.KeyVecDotQ4K,
        QuantizationConfig.KeyVecDotQ5K,
        QuantizationConfig.KeyVecDotQ6K,
        QuantizationConfig.KeyVecDotQ8_0,
        QuantizationConfig.KeyVecDotQ4_0,
        QuantizationConfig.KeyVecDotQ4_1,
        QuantizationConfig.KeyVecDotQ5_0,
        QuantizationConfig.KeyVecDotQ5_1,
        QuantizationConfig.KeyVecDotQ8_1,
        QuantizationConfig.KeyVecDotQ2K,
        QuantizationConfig.KeyVecDotQ8K,
        QuantizationConfig.KeyVecDotQ4_NL,
    ];

    private static readonly string[] QmmKeys = [
        QuantizationConfig.KeyQuantizedMatMulQ8_0,
        QuantizationConfig.KeyQuantizedMatMulQ5_0,
        QuantizationConfig.KeyQuantizedMatMulQ6K,
        QuantizationConfig.KeyQuantizedMatMulQ4_0,
        QuantizationConfig.KeyQuantizedMatMulQ4_1,
        QuantizationConfig.KeyQuantizedMatMulQ2K,
        QuantizationConfig.KeyQuantizedMatMulQ3K,
        QuantizationConfig.KeyQuantizedMatMulQ4K,
        QuantizationConfig.KeyQuantizedMatMulQ5K,
        QuantizationConfig.KeyQuantizedMatMulQ8K,
        QuantizationConfig.KeyQuantizedMatMulQ8_1,
        QuantizationConfig.KeyQuantizedMatMulQ5_1,
        QuantizationConfig.KeyQuantizedMatMulQ4_NL,
    ];

    public static MappingBuilder WithGpu(this MappingBuilder builder)
    {
        TryOverrideGpu(builder, SharpMindConfig.KeyPointWise);
        TryOverrideGpu(builder, SharpMindConfig.KeyGate);

        builder.Override(SharpMindConfig.KeySoftmax, GPUSharpMindConfig.ValSoftmax);
        builder.Override(SharpMindConfig.KeyRMSNorm, GPUSharpMindConfig.ValRMSNorm);

        foreach (var key in VecDotKeys)
            builder.Override(key, key + "_gpu");

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
