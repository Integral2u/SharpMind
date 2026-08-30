using ILGPU.Runtime;   // LoadEffectiveAddressAsPtr is an ArrayViewExtensions extension method
using SharpMind.GPU;
using Xunit;

namespace SharpMind.Tests.GPU;

[Collection("GPU")]
public sealed class DeviceArenaTests
{
    // ArrayView<T>.Index is not public in ILGPU 1.5.3, so the sub-view's start is
    // identified by its effective device address instead.
    [Fact]
    public unsafe void Rent_Reset_Rent_ReusesTheSameRegion()
    {
        using var arena = new DeviceArena(GpuTestDevice.Device, 1024);
        var t1 = arena.Rent(4, 8);
        Assert.Equal(32, arena.Used);
        arena.Reset();
        var t2 = arena.Rent(2, 2);
        Assert.Equal(4, arena.Used);
        Assert.Equal((IntPtr)t1.View.LoadEffectiveAddressAsPtr(), (IntPtr)t2.View.LoadEffectiveAddressAsPtr());
    }

    // A rented tensor is always a sub-view, and so is every slice of one. Zeroing must hit that
    // window and nothing around it.
    [Fact]
    public void Zero_ClearsOnlyTheSlicedRegion()
    {
        var dev = GpuTestDevice.Device;
        using var arena = new DeviceArena(dev, 1024);
        var before = arena.Rent(1, 4); before.Upload([9f, 9f, 9f, 9f]);
        var t = arena.Rent(2, 3); t.Upload([1f, 2f, 3f, 4f, 5f, 6f]);
        var after = arena.Rent(1, 4); after.Upload([7f, 7f, 7f, 7f]);

        t.Slice(1, 1).Zero();
        dev.Synchronize();

        Assert.Equal([1f, 2f, 3f, 0f, 0f, 0f], t.ToArray());
        Assert.Equal([9f, 9f, 9f, 9f], before.ToArray());
        Assert.Equal([7f, 7f, 7f, 7f], after.ToArray());
    }

    [Fact]
    public void Rent_BeyondCapacity_Throws()
    {
        using var arena = new DeviceArena(GpuTestDevice.Device, 16);
        arena.Rent(2, 8);
        Assert.Throws<InvalidOperationException>(() => arena.Rent(1, 1));
    }

    /// <summary>A non-positive dimension is rejected at the point of creation, not silently rented.</summary>
    [Theory]
    [InlineData(0, 4)]
    [InlineData(4, 0)]
    [InlineData(-1, 4)]
    [InlineData(4, -1)]
    public void Rent_RejectsNonPositiveDims(int rows, int cols)
    {
        using var arena = new DeviceArena(GpuTestDevice.Device, 1024);
        Assert.Throws<ArgumentOutOfRangeException>(() => arena.Rent(rows, cols));
    }

    /// <summary>
    /// DeviceTensor.Length is an int (as are the ILGPU launch extents), so a shape whose product
    /// overflows must be named here rather than surfacing as a negative length downstream.
    /// </summary>
    [Fact]
    public void Rent_RejectsAShapeThatOverflowsTheIntElementIndex()
    {
        using var arena = new DeviceArena(GpuTestDevice.Device, 1024);
        var ex = Assert.Throws<NotSupportedException>(() => arena.Rent(14_200, 151_936));
        Assert.Contains("151936", ex.Message);
    }
}
