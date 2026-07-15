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
        Tensor<float> projectionWeight,
        byte[]? rawWeight,
        QuantDType? rawDtype,
        Dictionary<string, string>? baseMapping)
    {
        var effectiveDtype = rawDtype ?? QuantDType.F32;

        if (baseMapping == null)
        {
            var fallbackDtype = QuantDType.F32;
            string fallbackPrefix = QuantHelper.DtypeToCompound(fallbackDtype);
            string fallbackCompound = $"{fallbackPrefix}_serial_scalar";
            var fallbackMapping = new Dictionary<string, string>
            {
                [SharpMindConfig.KeyLogit] = fallbackCompound
            };
            var fallbackType = _typeCache.GetOrAdd(
                MappingHash.Compute(fallbackMapping),
                _ => Assembler.Assemble<LogitOps>(fallbackMapping));
            return (LogitOps)Activator.CreateInstance(fallbackType, projectionWeight, rawWeight)!;
        }

        string compound = GetCompound(effectiveDtype, baseMapping);
        var mapping = new Dictionary<string, string>(baseMapping);
        mapping[SharpMindConfig.KeyLogit] = compound;

        var type = _typeCache.GetOrAdd(
            MappingHash.Compute(mapping),
            _ => Assembler.Assemble<LogitOps>(mapping));
        return (LogitOps)Activator.CreateInstance(type, projectionWeight, rawWeight)!;
    }

    internal static string GetCompound(QuantDType dtype, Dictionary<string, string> mapping)
    {
        string prefix = QuantHelper.DtypeToCompound(dtype);
        string? qmmSuffix = ExtractQmmSuffix(mapping);
        if (qmmSuffix == null)
        {
            string hw = System.Runtime.Intrinsics.X86.Fma.IsSupported ? "fma" :
                        System.Runtime.Intrinsics.X86.Avx2.IsSupported ? "avx2" :
                        System.Runtime.Intrinsics.X86.Sse3.IsSupported ? "sse" : "scalar";
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
