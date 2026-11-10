using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// A caller's mapping overrides are applied on top of the config's own mapping. The merge loops
/// used to write the config's value back for every key the config already had, so an override
/// could add a key but never change one: every kernel tier, activation and quant key a preset
/// sets was silently kept.
/// </summary>
public sealed class MappingOverrideTests
{
    private static readonly SharpMindConfig Scalar = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

    [Fact]
    public void ActivationFactory_OverrideOfAnExistingKey_ChangesTheKernel()
    {
        // Gpt is GELU; gelu(-1) is about -0.16, relu(-1) is exactly 0.
        var ops = ActivationFactory.Create(Scalar, new() { [SharpMindConfig.KeyPointWise] = SharpMindConfig.ValReLU });
        float[] dst = new float[1];
        ops.ApplyPointwise([-1f], dst);
        Assert.Equal(0f, dst[0]);
    }

    [Fact]
    public void GradientMappingFactory_OverrideOfAnExistingKey_ChangesTheMapping()
    {
        var plain = GradientMappingFactory.Create(Scalar, null);
        var overridden = GradientMappingFactory.Create(Scalar, new() { [SharpMindConfig.KeyGradLinear] = SharpMindConfig.ValAvx2 });
        Assert.Same(plain, GradientMappingFactory.Create(Scalar, null)); // instances are cached per mapping
        Assert.NotEqual(plain.GetType(), overridden.GetType()); // a different kernel set assembles a different type
    }

    [Fact]
    public void CreateTransformer_OverrideOfAnExistingKey_ReachesTheBlocks()
    {
        var cfg = new ModelConfig { VocabSize = 64, HiddenDim = 16, NumLayers = 1, NumHeads = 2, NumKvHeads = 2, FfnDim = 32, MaxSeqLen = 16 };
        using var weights = ModelFactory.CreateForTraining(cfg, Scalar);
        WeightInitializer.InitializeRandomly(weights, 1);

        // The preset is flash attention; the override picks the plain kernel for every block.
        var mapping = Scalar.ToJigSawMapping();
        mapping[SharpMindConfig.KeyAttention] = SharpMindConfig.ValMhaScalar;

        using var plain = ModelFactory.CreateTransformer(weights, Scalar, optimizeMemory: false);
        using var overridden = ModelFactory.CreateTransformer(weights, Scalar, mapping, optimizeMemory: false);
        Assert.NotEqual(plain.GetBlock(0)!.Attention!.GetType(), overridden.GetBlock(0)!.Attention!.GetType());
    }
}
