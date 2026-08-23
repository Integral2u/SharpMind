using SharpMind.Core.Memory;

namespace SharpMind.Tests.Memory;

public sealed class BigWorkspaceTests
{
    [Fact]
    public void Rent_And_Reset()
    {
        using var ws = new BigWorkspace(1024 * 1024);
        Assert.Equal(0, ws.UsedBytes);

        var shape = new ReadOnlySpan<int>([10, 10]);
        var t1 = ws.Rent<float>(shape);
        long usedAfterRent = ws.UsedBytes;
        Assert.True(usedAfterRent > 0);

        ws.Reset();
        Assert.Equal(0, ws.UsedBytes);
    }

    [Fact]
    public void Capacity_Reported()
    {
        using var ws = new BigWorkspace(2048);
        Assert.Equal(2048L, ws.CapacityBytes);
    }

    [Fact]
    public void UsagePercentage_Correct()
    {
        using var ws = new BigWorkspace(1024);
        Assert.Equal(0f, ws.UsagePercentage);

        ws.Rent<float>(new ReadOnlySpan<int>([256])); // 1024 bytes
        Assert.Equal(1f, ws.UsagePercentage);
    }

    [Fact]
    public void ExceedsCapacity_Throws()
    {
        using var ws = new BigWorkspace(128);
        Assert.Throws<OutOfMemoryException>(() =>
            ws.Rent<float>(new ReadOnlySpan<int>([128]))); // 512 bytes > 128
    }
}
