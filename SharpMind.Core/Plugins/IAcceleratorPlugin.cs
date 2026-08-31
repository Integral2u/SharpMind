namespace SharpMind.Core.Plugins;

/// <summary>
/// A hardware accelerator (GPU/TPU/NPU) supplied as a plugin DLL. Discovered by
/// <see cref="AcceleratorLoader"/> from the Plugins folder; the host never
/// references an accelerator assembly directly, so the main line stays pure .NET.
///
/// A plugin advertises what it can do through <see cref="Capabilities"/>, and the
/// host picks by type: <see cref="IMappingOverrides"/> for kernel-level (JigSaw)
/// substitution, <c>SharpMind.Training.ITrainingEngineFactory</c> for a resident
/// whole-step training engine, further capability types as they are defined.
///
/// Lifetime contract: implementations must have a parameterless constructor that
/// is cheap and side-effect free. The host constructs a plugin every time it scans
/// the plugins folder — the training wizard does so each time it opens — and never
/// disposes it, since this interface is not <see cref="IDisposable"/>. Acquire
/// device resources (a CUDA context, a driver handle, ...) in a capability such as
/// <c>ITrainingEngineFactory.TryCreate</c>, not in the constructor.
/// </summary>
public interface IAcceleratorPlugin
{
    /// <summary>
    /// Stable identifier ("ilgpu"); shown in the CUI and stored in training jobs. Case-insensitive.
    /// See <see cref="AcceleratorNames"/> for legacy-alias handling when a name is later renamed.
    /// </summary>
    string Name { get; }

    /// <summary>One line for humans: backend, requirements, what it accelerates.</summary>
    string Description { get; }

    /// <summary>Every capability this plugin offers. The host filters by type; unknown entries are ignored.</summary>
    IReadOnlyList<object> Capabilities { get; }
}
