using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using System.Collections.Concurrent;

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
        var layer = (InferenceLinearLayer)Activator.CreateInstance(type,
            name, inFeatures, outFeatures, bias, weight, biasTensor, quantDType)!;
        layer.WideAllowed = compound == Q8_0WideCompound || compound == Q4_KWideCompound;
        return layer;
    }

    /// <summary>The kernel selection <see cref="Q8_0WideWeights"/> stands in for.</summary>
    internal const string Q8_0WideCompound = "q8_0_parallel_fma";

    /// <summary>The kernel selection <see cref="Q4KWideWeights"/> stands in for.</summary>
    internal const string Q4_KWideCompound = "q4k_parallel_fma";

    public static LinearLayer Create(
        string name, int inFeatures, int outFeatures, bool bias,
        Tensor<float>? weight, Tensor<float>? biasTensor, QuantDType _)
    {
        return new TrainingLinearLayer(name, inFeatures, outFeatures, bias, weight, biasTensor);
    }

    internal static string GetCompound(QuantDType dtype, Dictionary<string, string> mapping)
        => QuantHelper.GetCompound(dtype, mapping);
}
