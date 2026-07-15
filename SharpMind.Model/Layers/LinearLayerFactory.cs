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

        string compound = QuantHelper.GetCompound(quantDType, baseMapping);
        var mapping = new Dictionary<string, string>(baseMapping);
        mapping[SharpMindConfig.KeyLinear] = compound;

        var type = _typeCache.GetOrAdd(MappingHash.Compute(mapping),
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
}
