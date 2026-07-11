namespace SharpMind.Model.Config;

/// <summary>
/// Extension methods on <see cref="ModelConfig"/> to create a <see cref="SharpMindConfig"/>.
/// </summary>
public static class SharpMindConfigFactory
{
    /// <summary>
    /// Creates the correct <see cref="SharpMindConfig"/> from a <see cref="ModelConfig"/>,
    /// using <see cref="ModelConfig.Architecture"/> to select the preset
    /// and dimensions to infer the attention variant.
    /// Applies any GGUF-sourced overrides (norm type, etc.) on top of the base preset.
    /// </summary>
    public static SharpMindConfig ForModel(this ModelConfig modelConfig, HardwareTier hw = HardwareTier.Auto)
    {
        var cfg = SharpMindConfig.ForModel(
            modelConfig.NumHeads,
            modelConfig.NumKvHeads,
            modelConfig.Architecture,
            hw);

        // Apply GGUF norm_type override (0=LayerNorm, 1=RMSNorm)
        if (modelConfig.NormTypeOverride.HasValue)
        {
            cfg = cfg with { Norm = modelConfig.NormTypeOverride.Value switch
            {
                0 => NormKind.LayerNorm,
                1 => NormKind.RMSNorm,
                _ => cfg.Norm
            }};
        }

        return cfg;
    }
}
