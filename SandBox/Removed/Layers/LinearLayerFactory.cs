using JigSawDotNet;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Format;
using System.Collections.Concurrent;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Model.Layers;

public abstract class LinearLayerFactory
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(LinearLayerKernels)}";

    [PuzzleCornerPiece(SharpMindConfig.KeyLinear,
        SharpMindConfig.ValLinearAuto,  $"{NS}.{nameof(LinearLayerKernels.CreateAuto)}",
        SharpMindConfig.ValLinearFloat, $"{NS}.{nameof(LinearLayerKernels.CreateFloat)}")]
    public abstract LinearLayer Create(string name, int inFeatures, int outFeatures, bool bias,
        QuantizationOps? qOps, Tensor<float>? weight, Tensor<float>? biasTensor,
        byte[]? rawData = null, GgufDtype? dtype = null);
}

public static class LinearLayerKernels
{
    private static readonly ConcurrentDictionary<string, Type> _qllTypeCache = [];

    public static LinearLayer CreateAuto(string name, int inFeatures, int outFeatures, bool bias,
        QuantizationOps? qOps, Tensor<float>? weight, Tensor<float>? biasTensor,
        byte[]? rawData = null, GgufDtype? dtype = null)
    {
        //return new OriginalLinearLayer(name, inFeatures, outFeatures, bias, qOps, weight, biasTensor);
        if (rawData != null)
        {
            var qtype = dtype ?? GgufDtype.F32;
            var qllType = GetQllType(qtype);
            return (QuantizedLinearLayer)Activator.CreateInstance(qllType,
                name, inFeatures, outFeatures, rawData, qtype, biasTensor)!;
        }
        return new FloatLinearLayer(name, inFeatures, outFeatures, bias, weight, biasTensor);
    }

    public static LinearLayer CreateFloat(string name, int inFeatures, int outFeatures, bool bias,
        QuantizationOps? qOps, Tensor<float>? weight, Tensor<float>? biasTensor,
        byte[]? rawData = null, GgufDtype? dtype = null)
    {

        //return new OriginalLinearLayer(name, inFeatures, outFeatures, bias, qOps, weight, biasTensor);
        return new FloatLinearLayer(name, inFeatures, outFeatures, bias, weight, biasTensor);
    }

    private static Type GetQllType(GgufDtype dtype)
    {
        string hw = Fma.IsSupported ? "fma" :
                     Avx2.IsSupported ? "avx2" :
                     Sse3.IsSupported ? "sse" : "scalar";
        string compound = DtypeToCompound(dtype, hw);
        return _qllTypeCache.GetOrAdd(compound, static c =>
        {
            return Assembler.Assemble<QuantizedLinearLayer>(new Dictionary<string, string>
            {
                [QuantizedLinearLayer.KeyQMatMul] = c
            });
        });
    }

    private static string DtypeToCompound(GgufDtype dtype, string hw)
    {
        string prefix = dtype switch
        {
            GgufDtype.Q8_0 => "q8_0",
            GgufDtype.Q4_0 => "q4_0",
            GgufDtype.Q4_1 => "q4_1",
            GgufDtype.Q5_0 => "q5_0",
            GgufDtype.Q5_1 => "q5_1",
            GgufDtype.Q8_1 => "q8_1",
            GgufDtype.IQ4_NL => "q4_nl",
            GgufDtype.Q2_K or GgufDtype.Q2_K_S => "q2k",
            GgufDtype.Q3_K or GgufDtype.Q3_K_S or GgufDtype.Q3_K_M or GgufDtype.Q3_K_L => "q3k",
            GgufDtype.Q4_K or GgufDtype.Q4_K_S or GgufDtype.Q4_K_M => "q4k",
            GgufDtype.Q5_K or GgufDtype.Q5_K_S or GgufDtype.Q5_K_M => "q5k",
            GgufDtype.Q6_K or GgufDtype.Q6_K_S => "q6k",
            GgufDtype.Q8_K => "q8k",
            GgufDtype.F32 => "f32",
            GgufDtype.F16 => "f16",
            _ => "f32"
        };
        return $"{prefix}_{hw}";
    }
}
