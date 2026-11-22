using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using System.Collections.Concurrent;

namespace SharpMind.Model;

public static class LogitOpsFactory
{
    private static readonly ConcurrentDictionary<int, Type> _typeCache = [];

    public static LogitOps Create(
        Tensor<float>? projectionWeight,
        byte[]? rawWeight,
        QuantDType? rawDtype,
        Dictionary<string, string>? baseMapping)
    {
        var effectiveDtype = rawDtype ?? QuantDType.F32;
        string compound = GetCompound(effectiveDtype, baseMapping);

        if (baseMapping == null)
        {
            var fallbackMapping = new Dictionary<string, string>
            {
                [SharpMindConfig.KeyLogit] = compound
            };
            var fallbackType = _typeCache.GetOrAdd(
                MappingHash.Compute(fallbackMapping),
                _ => Assembler.Assemble<LogitOps>(fallbackMapping));
            return WithCompound(
                (LogitOps)Activator.CreateInstance(fallbackType, projectionWeight, rawWeight)!,
                compound, rawDtype);
        }

        var mapping = new Dictionary<string, string>(baseMapping)
        {
            [SharpMindConfig.KeyLogit] = compound
        };

        var type = _typeCache.GetOrAdd(
            MappingHash.Compute(mapping),
            _ => Assembler.Assemble<LogitOps>(mapping));
        return WithCompound(
            (LogitOps)Activator.CreateInstance(type, projectionWeight, rawWeight)!,
            compound, rawDtype);
    }

    private static LogitOps WithCompound(LogitOps ops, string compound, QuantDType? rawDtype)
    {
        ops.WideAllowed = compound == Layers.LinearLayerFactory.Q8_0WideCompound
            || compound == Layers.LinearLayerFactory.Q4_KWideCompound
            || compound == Layers.LinearLayerFactory.Q6_KWideCompound;
        ops.RawDtype = rawDtype;
        return ops;
    }

    internal static string GetCompound(QuantDType dtype, Dictionary<string, string>? mapping)
        => QuantHelper.GetCompound(dtype, mapping);
}
