namespace SharpMind.Core.Plugins;

/// <summary>
/// Thrown by an accelerator resolver when a named plugin exists and offers the requested engine
/// factory, but the factory declined to create an engine (no CUDA/OpenCL device, unsupported model
/// shape, …). Carries the factory's human-readable <see cref="Reason"/>. The CUI turns this into a
/// consent dialog offering the CPU and every other capable plugin rather than failing the run or
/// launch outright; other lookups that genuinely can't be honoured (name not found, no factory)
/// remain plain <see cref="InvalidOperationException"/>s.
///
/// Derives from <see cref="InvalidOperationException"/> so callers that only catch that still work.
/// </summary>
public sealed class AcceleratorUnavailableException(string message, string reason, Exception? inner = null)
    : InvalidOperationException(message, inner)
{
    /// <summary>The factory's human-readable reason for declining (e.g. "no CUDA or OpenCL device is available on this machine.").</summary>
    public string Reason { get; } = reason;
}
