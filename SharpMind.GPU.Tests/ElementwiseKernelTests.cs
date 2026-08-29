using SharpMind.GPU;
using Xunit;

namespace SharpMind.GPU.Tests;

[Collection("GPU")]
public sealed class ElementwiseKernelTests
{
    [Fact]
    public void AddInPlace_AddBiasRows_Scale()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 10);
        var a = arena.Rent(2, 3); a.Upload([1, 2, 3, 4, 5, 6]);
        var b = arena.Rent(2, 3); b.Upload([10, 10, 10, 20, 20, 20]);
        var bias = arena.Rent(1, 3); bias.Upload([1, 0, -1]);
        dev.Kernels.AddInPlace(a, b);
        dev.Kernels.AddBiasRows(a, bias);
        dev.Kernels.Scale(a, 0.5f);
        dev.Synchronize();
        Assert.Equal([6f, 6f, 6f, 12.5f, 12.5f, 12.5f], a.ToArray());
    }

    [Fact]
    public void EmbedGather_CopiesSelectedRows()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 10);
        var table = arena.Rent(4, 2); table.Upload([0, 0, 1, 1, 2, 2, 3, 3]);
        using var ids = dev.UploadInts([3, 0, 2]);
        var x = arena.Rent(3, 2);
        dev.Kernels.EmbedGather(x, table, ids.View);
        dev.Synchronize();
        Assert.Equal([3f, 3f, 0f, 0f, 2f, 2f], x.ToArray());
    }

    [Fact]
    public void AddBiasRows_ThrowsOnMismatchedBiasLength()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 10);
        var x = arena.Rent(2, 3);
        var badBias = arena.Rent(1, 2);
        Assert.Throws<ArgumentException>(() => dev.Kernels.AddBiasRows(x, badBias));
    }

    [Fact]
    public void EmbedGather_ThrowsOnMismatchedIdsLength()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1 << 10);
        var table = arena.Rent(4, 2);
        using var ids = dev.UploadInts([0, 1]);
        var x = arena.Rent(3, 2);
        Assert.Throws<ArgumentException>(() => dev.Kernels.EmbedGather(x, table, ids.View));
    }
}
