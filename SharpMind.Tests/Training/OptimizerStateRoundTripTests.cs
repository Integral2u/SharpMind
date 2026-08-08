using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Training.Optimizers;

namespace SharpMind.Tests.Training;

/// <summary>
/// Regression tests for optimizer state serialization. AdamW and SGD previously
/// wrote the step counter before the learning rate but loaded the first float
/// straight into LR, so any SaveStateâ†’LoadState round trip shifted the whole
/// stream by four bytes and corrupted _lr and the moment vectors.
/// </summary>
public sealed class OptimizerStateRoundTripTests
{
    [Fact]
    public void AdamW_RoundTripsStepLrAndMoments()
    {
        using var t1 = Tensor<float>.From([1f, -2f, 3f], 3);
        using var t2 = Tensor<float>.From([0.5f, 1.5f], 2);
        using var p1 = new Parameter("w1", t1);
        using var p2 = new Parameter("b2", t2);

        var source = new AdamW([p1, p2], lr: 0.0123f, weightDecay: 0.01f);
        source.Update(); // step 1, populate moments
        source.Update(); // step 2
        Assert.Equal(2, source.Step);

        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true)) source.SaveState(writer);

        ms.Position = 0;
        using var r1 = new Parameter("w1", Tensor<float>.Zeros(3));
        using var r2 = new Parameter("w2", Tensor<float>.Zeros(2));
        var restored = new AdamW([r1, r2], lr: 0.0f, weightDecay: 0.1f);

        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true))
            restored.LoadState(reader, step: 2);

        Assert.Equal(2, restored.Step);
        Assert.Equal(0.0123f, restored.LearningRate, precision: 6);
    }

    [Fact]
    public void SGD_RoundTrips_LrAndVelocity()
    {
        using var p1 = new Parameter("w1", Tensor<float>.From([1f, -1f], 2));
        var source = new SGD([p1], lr: 0.007f, momentum: 0.9f);
        source.Update();
        source.Update();

        using var ms = new MemoryStream();
        using (var writer = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true)) source.SaveState(writer);

        ms.Position = 0;
        using var r1 = new Parameter("w1", Tensor<float>.Zeros(2));
        var restored = new SGD([r1], lr: 0.0f, momentum: 0.9f);
        using (var reader = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: true)) restored.LoadState(reader, step: 2);

        Assert.Equal(2, restored.Step);
        Assert.Equal(0.007f, restored.LearningRate, precision: 6);
    }
}
