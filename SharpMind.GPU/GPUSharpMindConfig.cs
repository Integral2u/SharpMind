using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using SharpMind.Core.Quantization;

namespace SharpMind.GPU;

public static class GPUSharpMindConfig
{
    // Mapping keys (linked to SharpMindConfig)
    public const string KeyPointWise = SharpMindConfig.KeyPointWise;
    public const string KeyGate = SharpMindConfig.KeyGate;
    public const string KeySoftmax = SharpMindConfig.KeySoftmax;
    public const string KeyRMSNorm = SharpMindConfig.KeyRMSNorm;
    public const string KeyMatMul = SharpMindConfig.KeyMatMul;

    // Mapping values (GPU suffix)
    public const string ValReLU = "relugpu";
    public const string ValGELU = "gelugpu";
    public const string ValSiLU = "silugpu";
    public const string ValSwiGLU = "swiglugpu";
    public const string ValGeGLU = "geglugpu";
    public const string ValSoftmax = "softmaxgpu";
    public const string ValRMSNorm = "rmsnormgpu";
    public const string ValMatMulNaive = "matmulgpunaive";
    public const string ValMatMulTiled = "matmulgputiled";
    public const string ValMatMulAdvanced = "matmulgpuadvanced";

    // Quantization GPU mapping values
    public const string ValVecDotQ3K = "q3k_gpu";
    public const string ValVecDotQ4K = "q4k_gpu";
    public const string ValVecDotQ5K = "q5k_gpu";
    public const string ValVecDotQ6K = "q6k_gpu";
    public const string ValVecDotQ8_0 = "q8_0_gpu";
    public const string ValVecDotQ4_0 = "q4_0_gpu";
    public const string ValVecDotQ4_1 = "q4_1_gpu";
    public const string ValVecDotQ5_0 = "q5_0_gpu";
    public const string ValVecDotQ5_1 = "q5_1_gpu";
    public const string ValVecDotQ8_1 = "q8_1_gpu";
    public const string ValVecDotQ2K = "q2k_gpu";
    public const string ValVecDotQ8K = "q8k_gpu";
    public const string ValHSum256 = "hsum_gpu";
    public const string ValHalfToFloat = "halftofloat_gpu";
    public const string ValGetScaleMinK4_Scale = "getscalemink4_scale_gpu";
    public const string ValGetScaleMinK4_Min = "getscalemink4_min_gpu";

    private static readonly Lazy<GPUMode> _backend = new(DetectBestBackend, LazyThreadSafetyMode.ExecutionAndPublication);

    public static GPUMode BestBackend => _backend.Value;

    public static bool HasGPU => BestBackend != GPUMode.Cpu;

    private static GPUMode DetectBestBackend()
    {
        using var ctx = Context.CreateDefault();
        if (ctx.GetCudaDevices().Count > 0) return GPUMode.Cuda;
        if (ctx.GetCLDevices().Count > 0) return GPUMode.OpenCL;
        return GPUMode.Cpu;
    }

    public static void AddGPUMappings(Dictionary<string, string> mapping)
    {
        mapping[SharpMindConfig.KeyPointWise] = ValReLU;
        mapping[SharpMindConfig.KeyGate] = ValGeGLU;
        mapping[SharpMindConfig.KeySoftmax] = ValSoftmax;
        mapping[SharpMindConfig.KeyRMSNorm] = ValRMSNorm;
        mapping[SharpMindConfig.KeyMatMul]  = ValMatMulNaive;
        mapping[QuantizationConfig.KeyVecDotQ3K]   = ValVecDotQ3K;
        mapping[QuantizationConfig.KeyVecDotQ4K]   = ValVecDotQ4K;
        mapping[QuantizationConfig.KeyVecDotQ5K]   = ValVecDotQ5K;
        mapping[QuantizationConfig.KeyVecDotQ6K]   = ValVecDotQ6K;
        mapping[QuantizationConfig.KeyVecDotQ8_0]  = ValVecDotQ8_0;
        mapping[QuantizationConfig.KeyVecDotQ4_0]  = ValVecDotQ4_0;
        mapping[QuantizationConfig.KeyVecDotQ4_1]  = ValVecDotQ4_1;
        mapping[QuantizationConfig.KeyVecDotQ5_0]  = ValVecDotQ5_0;
        mapping[QuantizationConfig.KeyVecDotQ5_1]  = ValVecDotQ5_1;
        mapping[QuantizationConfig.KeyVecDotQ8_1]  = ValVecDotQ8_1;
        mapping[QuantizationConfig.KeyVecDotQ2K]   = ValVecDotQ2K;
        mapping[QuantizationConfig.KeyVecDotQ8K]   = ValVecDotQ8K;
        mapping[QuantizationConfig.KeyHSum256]     = ValHSum256;
        mapping[QuantizationConfig.KeyHalfToFloat] = ValHalfToFloat;
        mapping[QuantizationConfig.KeyGetScaleMinK4_Scale] = ValGetScaleMinK4_Scale;
        mapping[QuantizationConfig.KeyGetScaleMinK4_Min]   = ValGetScaleMinK4_Min;
    }
}