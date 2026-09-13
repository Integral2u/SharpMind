using System;
using SharpMind.Core.Tensors;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// TrainingLinearLayer's forward hands the float kernel InFeatures × OutFeatures weights. A weight
/// tensor with fewer values was read past its end: the kernel faulted with an access violation that
/// crashed the whole test host when the next page happened to be unmapped, and silently produced
/// garbage when it did not. A mismatched weight must fail loudly and deterministically instead.
/// </summary>
public class TrainingLinearLayerShapeGuardTests
{
    [Fact]
    public void ForwardRejectsAWeightSmallerThanTheLayer()
    {
        using var layer = new TrainingLinearLayer("wgated_proj", 64, 256, bias: false,
            weight: new Tensor<float>(64, 128), biasTensor: null);
        using var input = new Tensor<float>(4, 64);

        var ex = Assert.Throws<InvalidOperationException>(() => layer.Forward(input));
        Assert.Contains("64 -> 256", ex.Message);
    }

    [Fact]
    public void ForwardAcceptsAMatchingWeight()
    {
        using var layer = new TrainingLinearLayer("wgated_proj", 64, 256, bias: false,
            weight: new Tensor<float>(64, 256), biasTensor: null);
        using var input = new Tensor<float>(4, 64);

        using var output = layer.Forward(input);
        Assert.Equal(256, output.Shape.Dims[^1]);
    }
}
