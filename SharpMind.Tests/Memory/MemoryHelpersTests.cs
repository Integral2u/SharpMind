using SharpMind.Core.Memory;

namespace SharpMind.Tests.Memory;

public sealed class MemoryHelpersTests
{
    [Fact]
    public void RentArray_ReturnsNonEmpty()
    {
        var arr = MemoryHelpers.RentArray<float>(100);
        Assert.NotNull(arr);
        Assert.True(arr.Length >= 100);
        MemoryHelpers.ReturnArray(arr);
    }

    [Fact]
    public void RentArray_ZeroLength_DoesNotThrow()
    {
        var arr = MemoryHelpers.RentArray<float>(0);
        Assert.NotNull(arr);
        MemoryHelpers.ReturnArray(arr);
    }

    [Fact]
    public void Rent_IntSize_ReturnsArray()
    {
        using var token = MemoryHelpers.Rent<float>(1000, out var intArray, out var bigArray);
        Assert.NotNull(intArray);
        Assert.Null(bigArray);
        Assert.True(intArray!.Length >= 1000);
    }

    [Fact]
    public void Rent_Oversized_ReturnsBigArray()
    {
        long count = (long)int.MaxValue / 2 + 100;
        using var token = MemoryHelpers.Rent<float>(count, out var intArray, out var bigArray);
        Assert.Null(intArray);
        Assert.NotNull(bigArray);
        Assert.Equal(count, bigArray!.Length);
    }

    [Fact]
    public void CreateWorkspace_CapacityUnder_ReturnsWorkspace()
    {
        using var ws = MemoryHelpers.CreateWorkspace(1024);
        Assert.IsType<Workspace>(ws);
    }

    [Fact]
    public void RentBuffer_ReturnsBuffer()
    {
        using var buf = MemoryHelpers.RentBuffer<float>(100);
        Assert.NotNull(buf);
        Assert.True(buf.Length >= 100);
        MemoryHelpers.ReturnBuffer(buf);
    }

    [Fact]
    public void MaxArrayPoolLength_IsReasonable()
    {
        Assert.True(MemoryHelpers.MaxArrayPoolLength > 0);
        Assert.True(MemoryHelpers.MaxArrayPoolLength < int.MaxValue);
    }
}
