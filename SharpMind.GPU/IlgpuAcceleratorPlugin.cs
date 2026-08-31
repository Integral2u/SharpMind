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
///
/// The plugin ships under the name <c>ilgpu</c> (<c>cuda</c> in earlier releases, kept as a
/// legacy alias by the resolvers so stored jobs/sessions still load). "ilgpu" not "cuda"
/// because it is the cross-vendor ILGPU backend: OpenCL devices work too, not just NVIDIA.
/// </summary>
public sealed class IlgpuAcceleratorPlugin : IAcceleratorPlugin
{
    /// <summary>Stored in a training job's <c>.smmt</c> and matched case-insensitively by the host.</summary>
    public string Name => "ilgpu";

    public string Description =>
        "GPU training via ILGPU kernels and cuBLAS GEMMs (NVIDIA CUDA) or ILGPU OpenCL. LoRA "
        + "fine-tuning of RMSNorm + RoPE/NoPE + gated-FFN models; requires a CUDA or OpenCL "
        + "device. Also provides GPU acceleration of the first inference prefill for the same "
        + "model shapes, with continued prefill and decode on the CPU (no incremental GPU decode "
        + "yet — see GpuInferenceEngine.DecodeStep).";

    public IReadOnlyList<object> Capabilities { get; } =
    [
        new GpuTrainingEngineFactory(),
        new GpuInferenceEngineFactory(),
        new GpuBackendHint(),
    ];
}

/// <summary>
/// Display capability: which backend the ILGPU path would run on this machine (OpenCL,
/// CUDA · cuBLAS or CUDA), or null when only the CPU fallback exists — which the plugin refuses.
/// The options screen appends this to the plugin's name so the user sees the real backend,
/// without the CUI referencing this assembly directly.
/// </summary>
public sealed class GpuBackendHint : IBackendHintProvider
{
    public string? BackendHint() => GpuDevice.BackendHint();
}
