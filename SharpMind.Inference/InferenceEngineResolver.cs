using SharpMind.Core.Plugins;

namespace SharpMind.Inference;

/// <summary>
/// Turns a session's accelerator choice into an <see cref="IInferenceEngine"/>.
/// Null means "the CPU default" — the host builds its usual generator. Any
/// request that cannot be honoured is an error, never a silent CPU fallback:
/// a user who asked for a GPU must not find out three hours later that they got
/// the CPU. Mirrors <see cref="SharpMind.Training.TrainingEngineResolver"/>.
/// </summary>
public static class InferenceEngineResolver
{
    /// <summary>The reserved name for the built-in CPU engine.</summary>
    public const string CpuName = "CPU";

    /// <summary>
    /// Resolves <paramref name="accelerator"/> (an <see cref="IAcceleratorPlugin.Name"/>,
    /// case-insensitive; null, blank or <see cref="CpuName"/> = CPU) against the
    /// loaded <paramref name="plugins"/>. The caller owns the returned engine.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// No plugin has that name, the plugin offers no <see cref="IInferenceEngineFactory"/>,
    /// or its factory declined (the message carries the factory's reason).
    /// </exception>
    public static IInferenceEngine? Resolve(string? accelerator, IReadOnlyList<IAcceleratorPlugin> plugins,
        InferenceEngineContext context)
    {
        ArgumentNullException.ThrowIfNull(plugins);
        ArgumentNullException.ThrowIfNull(context);

        if (string.IsNullOrWhiteSpace(accelerator) || accelerator.Trim().Equals(CpuName, StringComparison.OrdinalIgnoreCase))
            return null;

        string wanted = AcceleratorNames.Canonicalize(accelerator.Trim());
        var plugin = plugins.FirstOrDefault(p => string.Equals(p.Name, wanted, StringComparison.OrdinalIgnoreCase));
        if (plugin is null)
        {
            string available = plugins.Count == 0 ? "none loaded" : string.Join(", ", plugins.Select(p => p.Name));
            throw new InvalidOperationException($"Accelerator '{wanted}' was not found in the plugins folder (available: {available}).");
        }

        // Capabilities and TryCreate are third-party plugin code, called unprotected —
        // mirrors the same convention TrainingEngineResolver applies: a throw here must
        // not surface to the user as a bare, unattributed exception.
        IReadOnlyList<object> capabilities;
        try { capabilities = plugin.Capabilities; }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Accelerator '{plugin.Name}' failed while reading its capabilities: {ex.GetBaseException().Message}", ex);
        }

        var factory = capabilities.OfType<IInferenceEngineFactory>().FirstOrDefault()
            ?? throw new InvalidOperationException($"Accelerator '{plugin.Name}' does not provide an inference engine.");

        IInferenceEngine? engine;
        string? reason;
        try { engine = factory.TryCreate(context, out reason); }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                $"Accelerator '{plugin.Name}' failed while creating its inference engine: {ex.GetBaseException().Message}", ex);
        }

        return engine
            ?? throw new AcceleratorUnavailableException(
                $"Accelerator '{plugin.Name}' cannot infer this model: {reason ?? "no reason given"}.",
                reason ?? "no reason given");
    }
}