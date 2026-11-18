using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// MTP (NextN) GGUFs count their draft blocks in <c>block_count</c>, but llama.cpp
/// runs only <c>block_count - nextn_predict_layers</c> layers in the main pass —
/// an extra block there changes every logit. These pin the config arithmetic and
/// the loud failure when a file declares a head we cannot build.
/// </summary>
public sealed class MtpConfigTests
{
    private static ModelConfig BaseConfig => ModelConfig.Learnable with
    {
        Architecture = "qwen35",
        SsmConvKernel = 4,
        SsmInnerSize = 64,
        SsmStateSize = 8,
        SsmTimeStepRank = 4,
        SsmGroupCount = 2,
        FullAttentionInterval = 4,
    };

    [Fact]
    public void NextnLayers_AreCountedOutsideTheTrunk()
    {
        var config = BaseConfig with { NumLayers = 4, NextnPredictLayers = 1 };

        Assert.Equal(4, config.NumLayers);
        Assert.Equal(5, config.NumLayersTotal);
        Assert.False(config.IsMtpLayer(3));
        Assert.True(config.IsMtpLayer(4));
        Assert.False(config.IsMtpLayer(5));
    }

    /// <summary>
    /// The MTP block is dense attention even when its index lands on a recurrent
    /// slot of the hybrid pattern (llama.cpp flags NextN blocks non-recurrent).
    /// </summary>
    [Fact]
    public void MtpBlock_IsFullAttention_NotARecurrentSlot()
    {
        // Layer 4 would be recurrent by the (i+1) % 4 rule, but it is the MTP block.
        var config = BaseConfig with { NumLayers = 4, NextnPredictLayers = 1 };

        Assert.True(config.IsRecurrentLayer(2));
        Assert.False(config.IsRecurrentLayer(3));
        Assert.False(config.IsRecurrentLayer(4));
        Assert.True(config.IsFullAttentionLayer(4));
    }

    [Fact]
    public void Validation_RejectsMtpCountThatConsumesTheTrunk()
    {
        Assert.Throws<InvalidOperationException>(() =>
            (BaseConfig with { NumLayers = 2, NextnPredictLayers = 2 }).Validate());
        Assert.Throws<InvalidOperationException>(() =>
            (BaseConfig with { NumLayers = 2, NextnPredictLayers = -1 }).Validate());
    }

    [Fact]
    public void TrainingWeights_AllocateTheMtpBlock()
    {
        var config = BaseConfig with { NumLayers = 2, NextnPredictLayers = 1 };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

        using var weights = ModelFactory.CreateForTraining(config, sharpConfig);
        Assert.Equal(3, weights.Blocks.Length);
        Assert.Equal(2, config.NumLayers);
    }

    /// <summary>
    /// Declaring MTP without the head tensors must fail at assembly, not run with
    /// a silently missing draft head.
    /// </summary>
    [Fact]
    public void Transformer_WithDeclaredMtpButNoTensors_Throws()
    {
        var config = BaseConfig with { NumLayers = 2, NextnPredictLayers = 1 };
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        using var weights = ModelFactory.CreateForTraining(config, sharpConfig);

        Assert.Throws<InvalidDataException>(() => ModelFactory.CreateTransformer(weights, sharpConfig));
    }
}
