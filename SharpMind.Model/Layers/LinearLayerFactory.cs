using JigSawDotNet;
using SharpMind.Core;
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
        var mapping = new Dictionary<string, string>(baseMapping)
        {
            [SharpMindConfig.KeyLinear] = compound
        };

        var type = _typeCache.GetOrAdd(MappingHash.Compute(mapping),
            _ => Assembler.Assemble<InferenceLinearLayer>(mapping));
        return (InferenceLinearLayer)Activator.CreateInstance(type,
            name, inFeatures, outFeatures, bias, weight, biasTensor, quantDType)!;
    }

    public static LinearLayer Create(
        string name, int inFeatures, int outFeatures, bool bias,
        Tensor<float>? weight, Tensor<float>? biasTensor, QuantDType _)
    {
        return new TrainingLinearLayer(name, inFeatures, outFeatures, bias, weight, biasTensor);
    }

    internal static string GetCompound(QuantDType dtype, Dictionary<string, string> mapping)
    {
        string prefix = QuantHelper.DtypeToCompound(dtype);
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
        if (qmmVal == null) return null;

        int serialIdx = qmmVal.IndexOf("_serial_", StringComparison.Ordinal);
        int parallelIdx = qmmVal.IndexOf("_parallel_", StringComparison.Ordinal);
        int idx = serialIdx >= 0 ? serialIdx : parallelIdx;
        if (idx < 0) return null;

        return qmmVal.AsSpan(idx).ToString();
    }
}
