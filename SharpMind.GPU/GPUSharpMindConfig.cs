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
    public const string ValFloatToHalf = "floattohalf_gpu";
    public const string ValGetScaleMinK4_Scale = "getscalemink4_scale_gpu";
    public const string ValGetScaleMinK4_Min = "getscalemink4_min_gpu";

    // QuantizedMatMul GPU mapping values (serial)
    public const string ValQuantizedMatMulQ8_0_Serial = "qmatmul_q8_0_serial_gpu";
    public const string ValQuantizedMatMulQ5_0_Serial = "qmatmul_q5_0_serial_gpu";
    public const string ValQuantizedMatMulQ6K_Serial  = "qmatmul_q6k_serial_gpu";
    public const string ValQuantizedMatMulQ4_0_Serial = "qmatmul_q4_0_serial_gpu";
    public const string ValQuantizedMatMulQ4_1_Serial = "qmatmul_q4_1_serial_gpu";
    public const string ValQuantizedMatMulQ2K_Serial  = "qmatmul_q2k_serial_gpu";
    public const string ValQuantizedMatMulQ3K_Serial  = "qmatmul_q3k_serial_gpu";
    public const string ValQuantizedMatMulQ4K_Serial  = "qmatmul_q4k_serial_gpu";
    public const string ValQuantizedMatMulQ5K_Serial  = "qmatmul_q5k_serial_gpu";
    public const string ValQuantizedMatMulQ8K_Serial  = "qmatmul_q8k_serial_gpu";
    public const string ValQuantizedMatMulQ8_1_Serial = "qmatmul_q8_1_serial_gpu";
    public const string ValQuantizedMatMulQ5_1_Serial = "qmatmul_q5_1_serial_gpu";

    // QuantizedMatMul GPU mapping values (parallel)
    public const string ValQuantizedMatMulQ8_0_Parallel = "qmatmul_q8_0_parallel_gpu";
    public const string ValQuantizedMatMulQ5_0_Parallel = "qmatmul_q5_0_parallel_gpu";
    public const string ValQuantizedMatMulQ6K_Parallel  = "qmatmul_q6k_parallel_gpu";
    public const string ValQuantizedMatMulQ4_0_Parallel = "qmatmul_q4_0_parallel_gpu";
    public const string ValQuantizedMatMulQ4_1_Parallel = "qmatmul_q4_1_parallel_gpu";
    public const string ValQuantizedMatMulQ2K_Parallel  = "qmatmul_q2k_parallel_gpu";
    public const string ValQuantizedMatMulQ3K_Parallel  = "qmatmul_q3k_parallel_gpu";
    public const string ValQuantizedMatMulQ4K_Parallel  = "qmatmul_q4k_parallel_gpu";
    public const string ValQuantizedMatMulQ5K_Parallel  = "qmatmul_q5k_parallel_gpu";
    public const string ValQuantizedMatMulQ8K_Parallel  = "qmatmul_q8k_parallel_gpu";
    public const string ValQuantizedMatMulQ8_1_Parallel = "qmatmul_q8_1_parallel_gpu";
    public const string ValQuantizedMatMulQ5_1_Parallel = "qmatmul_q5_1_parallel_gpu";

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
        mapping[QuantizationKeys.KeyVecDotQ3K]   = ValVecDotQ3K;
        mapping[QuantizationKeys.KeyVecDotQ4K]   = ValVecDotQ4K;
        mapping[QuantizationKeys.KeyVecDotQ5K]   = ValVecDotQ5K;
        mapping[QuantizationKeys.KeyVecDotQ6K]   = ValVecDotQ6K;
        mapping[QuantizationKeys.KeyVecDotQ8_0]  = ValVecDotQ8_0;
        mapping[QuantizationKeys.KeyVecDotQ4_0]  = ValVecDotQ4_0;
        mapping[QuantizationKeys.KeyVecDotQ4_1]  = ValVecDotQ4_1;
        mapping[QuantizationKeys.KeyVecDotQ5_0]  = ValVecDotQ5_0;
        mapping[QuantizationKeys.KeyVecDotQ5_1]  = ValVecDotQ5_1;
        mapping[QuantizationKeys.KeyVecDotQ8_1]  = ValVecDotQ8_1;
        mapping[QuantizationKeys.KeyVecDotQ2K]   = ValVecDotQ2K;
        mapping[QuantizationKeys.KeyVecDotQ8K]   = ValVecDotQ8K;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ8_0] = ValQuantizedMatMulQ8_0_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ5_0] = ValQuantizedMatMulQ5_0_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ6K]  = ValQuantizedMatMulQ6K_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ4_0] = ValQuantizedMatMulQ4_0_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ4_1] = ValQuantizedMatMulQ4_1_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ2K]  = ValQuantizedMatMulQ2K_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ3K]  = ValQuantizedMatMulQ3K_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ4K]  = ValQuantizedMatMulQ4K_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ5K]  = ValQuantizedMatMulQ5K_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ8K]  = ValQuantizedMatMulQ8K_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ8_1] = ValQuantizedMatMulQ8_1_Serial;
        mapping[QuantizationKeys.KeyQuantizedMatMulQ5_1] = ValQuantizedMatMulQ5_1_Serial;
        mapping[QuantizationKeys.KeyHSum256]     = ValHSum256;
        mapping[QuantizationKeys.KeyHalfToFloat] = ValHalfToFloat;
        mapping[QuantizationKeys.KeyFloatToHalf] = ValFloatToHalf;
        mapping[QuantizationKeys.KeyGetScaleMinK4_Scale] = ValGetScaleMinK4_Scale;
        mapping[QuantizationKeys.KeyGetScaleMinK4_Min]   = ValGetScaleMinK4_Min;
    }
}