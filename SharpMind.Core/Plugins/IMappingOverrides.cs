namespace SharpMind.Core.Plugins;

/// <summary>
/// Kernel-level accelerator capability: JigSaw mapping entries (key → value) that
/// replace the CPU kernels chosen by a <see cref="SharpMindConfig"/> preset. The
/// host applies them on top of the preset before assembling ops.
/// </summary>
public interface IMappingOverrides
{
    /// <summary>Overrides for <paramref name="config"/>. May be empty; never null.</summary>
    IReadOnlyDictionary<string, string> GetOverrides(SharpMindConfig config);
}
