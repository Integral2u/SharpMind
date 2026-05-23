namespace SharpMind.Model.Config;

/// <summary>
/// Extension methods on <see cref="ModelConfig"/> to create a <see cref="global::SharpMind.SharpMindConfig"/>.
/// </summary>
public static class SharpMindConfigFactory
{
    /// <summary>
    /// Creates the correct <see cref="global::SharpMind.SharpMindConfig"/> from a <see cref="ModelConfig"/>,
    /// using <see cref="ModelConfig.Architecture"/> to select the preset
    /// and dimensions to infer the attention variant.
    /// </summary>
    public static global::SharpMind.SharpMindConfig ForModel(this ModelConfig modelConfig, global::SharpMind.HardwareTier hw = global::SharpMind.HardwareTier.Auto)
    {
        return global::SharpMind.SharpMindConfig.ForModel(
            modelConfig.NumHeads,
            modelConfig.NumKvHeads,
            modelConfig.Architecture,
            hw);
    }
}
