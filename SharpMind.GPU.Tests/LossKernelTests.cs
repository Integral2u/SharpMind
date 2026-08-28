using SharpMind.Core.Tensors;
using SharpMind.GPU;
using SharpMind.Training.Autograd;
using SharpMind.Training.Loss;
using Xunit;

namespace SharpMind.GPU.Tests;

[Collection("GPU")]
public sealed class LossKernelTests
{
    const int T = 6, V = 40;

    [Theory] [InlineData(0f)] [InlineData(0.1f)]
    public void CeLossAndGrad_MatchesCpuLoss(float smoothing)
    {
        var dev = GpuTestDevice.Device;
        var logits = GpuTestDevice.Random(T * V, 31, 3f);
        int[] labels = [3, 7, -100, 0, 39, 12];          // one ignored row
        using var cpuLogits = Tensor<float>.From(logits, T, V);
        using var cpuLabels = Tensor<int>.From(labels, T);
        var loss = new CrossEntropyLoss(labelSmoothing: smoothing);
        float wantLoss = loss.Compute(cpuLogits, cpuLabels);
        using var wantGrad = Gradients.CrossEntropySoftmax(cpuLogits, cpuLabels, -100, smoothing);

        using var arena = new DeviceArena(dev, 1 << 12);
        var tl = arena.Rent(T, V); tl.Upload(logits);
        var rowLoss = arena.Rent(T, 1);
        using var ids = dev.UploadInts(labels);
        float got = dev.Kernels.CeLossAndGrad(tl, ids.View, labels, rowLoss, -100, smoothing);
        Assert.Equal(wantLoss, got, 5);
        GpuTestDevice.AssertClose(wantGrad.Data, tl.ToArray(), 1e-5, "dLogits");
    }

    [Fact]
    public void CeLossAndGrad_RejectsWrongRowLossShape()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 10);
        var logits = arena.Rent(T, V);
        var badRowLoss = arena.Rent(T, 2);
        int[] labels = [0, 1, 2, 3, 4, 5];
        using var ids = dev.UploadInts(labels);
        Assert.Throws<ArgumentException>(() => dev.Kernels.CeLossAndGrad(logits, ids.View, labels, badRowLoss, -100, 0f));
    }

    [Fact]
    public void CeLossAndGrad_RejectsLabelsLengthMismatch()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 10);
        var logits = arena.Rent(T, V);
        var rowLoss = arena.Rent(T, 1);
        int[] labels = [0, 1, 2, 3, 4]; // T-1
        using var ids = dev.UploadInts(labels);
        Assert.Throws<ArgumentException>(() => dev.Kernels.CeLossAndGrad(logits, ids.View, labels, rowLoss, -100, 0f));
    }

    [Fact]
    public void CeLossAndGrad_RejectsHostLabelsLengthMismatch()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 10);
        var logits = arena.Rent(T, V);
        var rowLoss = arena.Rent(T, 1);
        int[] labels = [0, 1, 2, 3, 4, 5];
        int[] hostLabels = [0, 1, 2, 3, 4]; // T-1
        using var ids = dev.UploadInts(labels);
        Assert.Throws<ArgumentException>(() => dev.Kernels.CeLossAndGrad(logits, ids.View, hostLabels, rowLoss, -100, 0f));
    }

    [Theory] [InlineData(-0.1f)] [InlineData(1f)]
    public void CeLossAndGrad_RejectsInvalidLabelSmoothing(float smoothing)
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 10);
        var logits = arena.Rent(T, V);
        var rowLoss = arena.Rent(T, 1);
        int[] labels = [0, 1, 2, 3, 4, 5];
        using var ids = dev.UploadInts(labels);
        Assert.Throws<ArgumentException>(() => dev.Kernels.CeLossAndGrad(logits, ids.View, labels, rowLoss, -100, smoothing));
    }
}
