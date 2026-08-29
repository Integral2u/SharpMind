using SharpMind.Core.Plugins;

namespace SharpMind.GPU;

/// <summary>
/// Advertises the ILGPU/cuBLAS training engine to the host. Discovered by
/// <c>AcceleratorLoader</c> from the Plugins folder; nothing in the application references
/// this assembly.
///
/// The constructor deliberately does nothing. The host builds one of these every time it scans
/// the plugins folder and never disposes it, so the device is acquired in
/// <see cref="GpuTrainingEngineFactory.TryCreate"/> instead, where it has an owner.
/// </summary>
public sealed class CudaAcceleratorPlugin : IAcceleratorPlugin
{
    /// <summary>Stored in a training job's <c>.smmt</c> and matched case-insensitively by the host.</summary>
    public string Name => "cuda";

    public string Description =>
        "NVIDIA GPU training via ILGPU kernels and cuBLAS GEMMs. LoRA fine-tuning of "
        + "RMSNorm + RoPE/NoPE + gated-FFN models; requires a CUDA or OpenCL device.";

    public IReadOnlyList<object> Capabilities { get; } = [new GpuTrainingEngineFactory()];
}
