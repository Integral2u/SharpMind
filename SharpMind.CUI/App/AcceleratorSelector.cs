using SharpMind.Core.Plugins;

namespace SharpMind.CUI.App;

/// <summary>
/// Shared plumbing for the "Accelerator:" selectors in <c>OptionsView</c> (inference) and
/// <c>TrainingWizardView</c> (training). Both show "CPU" plus every discovered plugin that offers
/// the matching engine capability (so a plugin that can't do the job doesn't appear as selectable),
/// and both resolve legacy names (e.g. a pre-rename stored <c>"cuda"</c> -&gt; <c>"ilgpu"</c>) so a
/// saved job/preset still selects the right row and is rewritten to the canonical name on save.
/// </summary>
public static class AcceleratorSelector
{
    /// <summary>The reserved name for the built-in CPU engine (matches both resolvers' <c>CpuName</c>).</summary>
    public const string CpuName = "CPU";

    /// <summary>Accelerator label list ("CPU" + plugin names offering <paramref name="capabilityType"/>) for a selector.</summary>
    public static string[] LabelNames(string? pluginsFolder, Type capabilityType)
    {
        var plugins = AcceleratorLoader.LoadFrom(pluginsFolder, out _);
        var names = new List<string> { CpuName };
        foreach (var p in plugins)
        {
            bool offers;
            try { offers = p.Capabilities.Any(c => capabilityType.IsInstanceOfType(c)); }
            catch { offers = false; }
            if (!offers) continue;
            // A plugin may carry an IBackendHintProvider capability naming the backend it would
            // actually run (OpenCL / CUDA·cuBLAS / CUDA) on this machine. Shown as a "(...)" hint so
            // the user sees the real backend, not a bare plugin name; null means no real device so the
            // row stays bare.
            names.Add(BuildLabel(p, capabilityType));
        }
        return [.. names];
    }

    private static string BuildLabel(IAcceleratorPlugin plugin, Type capabilityType)
    {
        string? hint = null;
        try
        {
            foreach (var c in plugin.Capabilities)
                if (c is IBackendHintProvider p) { hint = p.BackendHint(); if (!string.IsNullOrEmpty(hint)) break; }
        }
        catch { hint = null; }
        return string.IsNullOrEmpty(hint) ? plugin.Name : $"{plugin.Name} ({hint})";
    }

    /// <summary>
    /// The stored accelerator value for a selector label: strips the <c>"(OpenCL)"</c> style display
    /// hint off <paramref name="label"/>, returning the canonical plugin name (or <see cref="CpuName"/>
    /// for the CPU row). The hint is presentation only — the value that gets stored in a job/preset and
    /// matched by the resolvers is the bare name.
    /// </summary>
    public static string ValueOf(string? label)
    {
        string s = (label ?? "").Trim();
        int open = s.LastIndexOf(" (", StringComparison.Ordinal);
        return open > 0 && s.EndsWith(')') ? s[..open] : s;
    }

    /// <summary>
    /// Index into <paramref name="labels"/> for a (possibly legacy) accelerator value: maps a stored
    /// <c>"cuda"</c> to the <c>"ilgpu"</c> row, 0 (CPU) for null/blank/unknown. Matches rows by their
    /// stored value (<see cref="ValueOf"/>), so a display hint on the row does not break the lookup.
    /// </summary>
    public static int IndexFor(string[] labels, string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return 0;
        string canonical = AcceleratorNames.Canonicalize(value.Trim());
        int i = Array.FindIndex(labels, n => AcceleratorNames.Canonicalize(ValueOf(n)).Equals(canonical, StringComparison.OrdinalIgnoreCase));
        return i < 0 ? 0 : i;
    }
}
