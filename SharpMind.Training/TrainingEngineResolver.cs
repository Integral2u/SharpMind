using SharpMind.Core.Plugins;

namespace SharpMind.Training;

/// <summary>
/// Turns a training job's accelerator choice into an <see cref="ITrainingEngine"/>.
/// Null means "the CPU default" — <see cref="TrainLoop"/> builds its own. Any
/// request that cannot be honoured is an error, never a silent CPU fallback: a
/// user who asked for a GPU must not find out three hours later that they got
/// the CPU.
/// </summary>
public static class TrainingEngineResolver
{
    /// <summary>The reserved name for the built-in CPU engine.</summary>
    public const string CpuName = "CPU";

    /// <summary>
    /// Resolves <paramref name="accelerator"/> (an <see cref="IAcceleratorPlugin.Name"/>,
    /// case-insensitive; null, blank or <see cref="CpuName"/> = CPU) against the
    /// loaded <paramref name="plugins"/>. The caller owns the returned engine.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// No plugin has that name, the plugin offers no <see cref="ITrainingEngineFactory"/>,
    /// or its factory declined (the message carries the factory's reason).
    /// </exception>
    public static ITrainingEngine? Resolve(string? accelerator, IReadOnlyList<IAcceleratorPlugin> plugins, TrainingEngineContext context)
    {
        ArgumentNullException.ThrowIfNull(plugins);
        ArgumentNullException.ThrowIfNull(context);

        if (string.IsNullOrWhiteSpace(accelerator) || accelerator.Trim().Equals(CpuName, StringComparison.OrdinalIgnoreCase))
            return null;

        string wanted = accelerator.Trim();
        var plugin = plugins.FirstOrDefault(p => string.Equals(p.Name, wanted, StringComparison.OrdinalIgnoreCase));
        if (plugin is null)
        {
            string available = plugins.Count == 0 ? "none loaded" : string.Join(", ", plugins.Select(p => p.Name));
            throw new InvalidOperationException($"Accelerator '{wanted}' was not found in the plugins folder (available: {available}).");
        }

        var factory = plugin.Capabilities.OfType<ITrainingEngineFactory>().FirstOrDefault()
            ?? throw new InvalidOperationException($"Accelerator '{plugin.Name}' does not provide a training engine.");

        var engine = factory.TryCreate(context, out var reason);
        return engine
            ?? throw new InvalidOperationException($"Accelerator '{plugin.Name}' cannot train this job: {reason ?? "no reason given"}.");
    }
}
