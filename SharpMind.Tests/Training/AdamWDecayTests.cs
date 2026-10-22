using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Training.Optimizers;
using Xunit;

namespace SharpMind.Tests.Training;

public sealed class AdamWDecayTests
{
    [Fact]
    public void NormLayerWeights_AreExcludedFromWeightDecay()
    {
        using var normalData = Tensor<float>.Ones(2);
        using var rmsData = Tensor<float>.Ones(2);
        using var layerData = Tensor<float>.Ones(2);
        using var normal = new Parameter("blocks.0.attention.Wq.weight", normalData);
        using var rms = new Parameter("RmsNormLayer.weight", rmsData);
        using var layer = new Parameter("LayerNormLayer.weight", layerData);

        using var optimizer = new AdamW(
            [normal, rms, layer], lr: 0.1f, weightDecay: 0.1f);

        optimizer.Update();

        // Zero gradient: Adam's adaptive term is zero, so only decay moves a weight.
        Assert.True(normalData.Data[0] < 1f, "ordinary weights should receive decay");
        Assert.Equal(1f, rmsData.Data[0], precision: 6);
        Assert.Equal(1f, layerData.Data[0], precision: 6);
    }
}
