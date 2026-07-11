using JigSawDotNet;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using System.Collections.Concurrent;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Model.Layers;

public static class LinearLayerFactory
{
    private static readonly ConcurrentDictionary<int, Type> _typeCache = [];

    public static LinearLayer Create(
        string name, int inFeatures, int outFeatures, bool bias,
        Tensor<float>? weight, Tensor<float>? biasTensor, QuantDType quantDType,
        Dictionary<string, string>? baseMapping)
    {
        if (baseMapping == null)
            return new TrainingLinearLayer(name, inFeatures, outFeatures, bias, weight, biasTensor);

        string compound = GetCompound(quantDType, baseMapping);
        var mapping = new Dictionary<string, string>(baseMapping);
        mapping[SharpMindConfig.KeyLinear] = compound;

        var type = _typeCache.GetOrAdd(mapping.GetHashCode(),
            _ => Assembler.Assemble<InferenceLinearLayer>(mapping));
        return (InferenceLinearLayer)Activator.CreateInstance(type,
            name, inFeatures, outFeatures, bias, weight, biasTensor, quantDType)!;
    }

    public static LinearLayer Create(
        string name, int inFeatures, int outFeatures, bool bias,
        Tensor<float>? weight, Tensor<float>? biasTensor, QuantDType quantDType)
    {
        return new TrainingLinearLayer(name, inFeatures, outFeatures, bias, weight, biasTensor);
    }

    internal static string GetCompound(QuantDType dtype, Dictionary<string, string> mapping)
    {
        string prefix = DtypeToCompound(dtype);
        string? qmmSuffix = ExtractQmmSuffix(mapping);
        if (qmmSuffix == null)
        {
            string hw = Fma.IsSupported ? "fma" :
                        Avx2.IsSupported ? "avx2" :
                        Sse3.IsSupported ? "sse" : "scalar";
            qmmSuffix = $"_serial_{hw}";
        }
        return $"{prefix}{qmmSuffix}";
    }

    private static string? ExtractQmmSuffix(Dictionary<string, string> mapping)
    {
        string? qmmVal = mapping.GetValueOrDefault("qmatmul_f32");
        if (qmmVal != null)
        {
            int idx = qmmVal.IndexOf('_', StringComparison.Ordinal);
            if (idx >= 0)
            {
                string suffix = qmmVal.AsSpan(idx).ToString();
                if (suffix.StartsWith("_serial_", StringComparison.Ordinal) || suffix.StartsWith("_parallel_", StringComparison.Ordinal))
                    return suffix;
            }
        }
        return null;
    }

    private static string DtypeToCompound(QuantDType dtype) => dtype switch
    {
        QuantDType.Q8_0 => "q8_0",
        QuantDType.Q4_0 => "q4_0",
        QuantDType.Q4_1 => "q4_1",
        QuantDType.Q5_0 => "q5_0",
        QuantDType.Q5_1 => "q5_1",
        QuantDType.Q8_1 => "q8_1",
        QuantDType.Q2_K => "q2k",
        QuantDType.Q3_K => "q3k",
        QuantDType.Q4_K => "q4k",
        QuantDType.Q5_K => "q5k",
        QuantDType.Q6_K => "q6k",
        QuantDType.Q8_K => "q8k",
        QuantDType.IQ4_NL => "q4_nl",
        QuantDType.F16 => "f16",
        _ => "f32"
    };
}
