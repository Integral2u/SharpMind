using ILGPU;
using ILGPU.Runtime.Cuda;
using ILGPU.Runtime.OpenCL;

namespace SharpMind.GPU;

public enum GPUMode
{
    Cpu,
    OpenCL,
    Cuda
}

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
    public const string ValMatMul = "matmulgpu";

    private static GPUMode? _backend;

    public static GPUMode BestBackend => _backend ??= DetectBestBackend();

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
        mapping[KeyPointWise] = ValReLU;
        mapping[KeyGate] = ValGeGLU;
        mapping[KeySoftmax] = ValSoftmax;
        mapping[KeyRMSNorm] = ValRMSNorm;
        mapping[KeyMatMul]  = ValMatMul;
    }
}