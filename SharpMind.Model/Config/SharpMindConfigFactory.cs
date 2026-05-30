namespace SharpMind.Model.Config;

/// <summary>
/// Extension methods on <see cref="ModelConfig"/> to create a <see cref="global::SharpMind.SharpMindConfig"/>.
/// </summary>
public static class SharpMindConfigFactory
{
    /// <summary>
    /// Creates the correct <see cref="SharpMindConfig"/> from a <see cref="ModelConfig"/>,
    /// using <see cref="ModelConfig.Architecture"/> to select the preset
    /// and dimensions to infer the attention variant.
    /// </summary>
    public static SharpMindConfig ForModel(this ModelConfig modelConfig, HardwareTier hw = HardwareTier.Auto)
    {
        return SharpMindConfig.ForModel(
            modelConfig.NumHeads,
            modelConfig.NumKvHeads,
            modelConfig.Architecture,
            hw);
    }
}
