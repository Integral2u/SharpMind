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
            || compound == Layers.LinearLayerFactory.Q6_KWideCompound
            || compound == "q4_0_parallel_fma"
            || compound == "q4_1_parallel_fma"
            || compound == "q5_0_parallel_fma"
            || compound == "q5_1_parallel_fma"
            || compound == "q4_nl_parallel_fma"
            || compound == "q5k_parallel_fma"
            || compound == "q2k_parallel_fma"
            || compound == "q3k_parallel_fma"
            || compound == "iq4_xs_parallel_fma"
            || compound == "iq3_xxs_parallel_fma"
            || compound == "iq2_xxs_parallel_fma"
            || compound == "iq2_xs_parallel_fma"
            || compound == "iq2_s_parallel_fma"
            || compound == "iq3_s_parallel_fma"
            || compound == "mxfp4_parallel_fma";
        ops.RawDtype = rawDtype;
        return ops;
    }

    internal static string GetCompound(QuantDType dtype, Dictionary<string, string>? mapping)
        => QuantHelper.GetCompound(dtype, mapping);
}
