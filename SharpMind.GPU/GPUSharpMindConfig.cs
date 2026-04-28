using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;
using System.Reflection.Metadata;

namespace SharpMind.GPU;

/// <summary>
/// Hardware backend for ILGPU acceleration.
/// Preference order: CUDA → OpenCL → CPU.
/// </summary>
public enum GPUMode
{
    Cpu,
    OpenCL,
    Cuda
}

/// <summary>
/// GPU-accelerated configuration helpers.
/// Mirrors SharpMindConfig value conventions with "ilgpu" suffixes.
/// </summary>
public static class GPUSharpMindConfig
{
    // Mapping keys
    public const string MapActivationKeyPointWise = SharpMindConfig.MapActivationKeyPointWise;
    public const string MapActivationKeyGate = SharpMindConfig.MapActivationKeyGate;
    public const string MapActivationKeySoftMax = SharpMindConfig.MapActivationKeySoftMax;
    public const string MapActivationKeyRMSNorm = SharpMindConfig.MapActivationKeyRMSNorm;
    public const string MapActivationKeyMatMul = SharpMindConfig.MapActivationKeyMatMul;

    // Mapping values (ILGPU suffix)
    public const string MapActivationKernelReLU = "relugpu";
    public const string MapActivationKernelGELU = "gelugpu";
    public const string MapActivationKernelSiLU = "silugpu";
    public const string MapActivationKernelSwiGLU = "swiglutilgpu";
    public const string MapActivationKernelGeGLU = "geglugpu";
    public const string MapActivationKernelSoftMax = "softmaxgpu";
    public const string MapActivationKernelRMSNorm = "rmsnormgpu";
    public const string MapActivationKernelMatMul = "matmulgpu";

    private static GPUMode? _backend;

    public static GPUMode BestBackend => _backend ??= DetectBestBackend();

    public static bool HasGPU => BestBackend != GPUMode.Cpu;

    private static GPUMode DetectBestBackend()
    {
        // Check for CUDA/OpenCL availability
        using var ctx = Context.CreateDefault();
        if (ctx.GetCudaDevices().Count > 0) return GPUMode.Cuda;
        if (ctx.GetCLDevices().Count > 0) return GPUMode.OpenCL;
        return GPUMode.Cpu;
    }

    public static void AddGPUMappings(Dictionary<string, string> mapping)
    {
        mapping[MapActivationKeyPointWise] = MapActivationKernelReLU;
        mapping[MapActivationKeyGate] = MapActivationKernelGeGLU;
        mapping[MapActivationKeySoftMax] = MapActivationKernelSoftMax;
        mapping[MapActivationKeyRMSNorm] = MapActivationKernelRMSNorm;
        mapping[MapActivationKeyMatMul]  = MapActivationKernelMatMul;
    }
}