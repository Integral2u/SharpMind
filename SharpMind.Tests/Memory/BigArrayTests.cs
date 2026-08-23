using SharpMind.Core.Memory;

namespace SharpMind.Tests.Memory;

public sealed class BigArrayTests
{
    [Fact]
    public void Length_ReturnsCorrectCount()
    {
        using var arr = new BigArray<float>(1_000_000);
        Assert.Equal(1_000_000L, arr.Length);
    }

    [Fact]
    public void Indexer_ReadWrite_SinglePage()
    {
        using var arr = new BigArray<float>(10);
        arr[0] = 1.0f;
        arr[9] = 9.0f;
        Assert.Equal(1.0f, arr[0]);
        Assert.Equal(9.0f, arr[9]);
    }

    [Fact]
    public void Indexer_ReadWrite_CrossPage()
    {
        int pageSize = 1024 * 1024; // 1M
        using var arr = new BigArray<float>(pageSize + 10);
        arr[0] = 0.1f;
        arr[pageSize - 1] = 0.2f;
        arr[pageSize] = 0.3f;
        arr[pageSize + 9] = 0.4f;

        Assert.Equal(0.1f, arr[0]);
        Assert.Equal(0.2f, arr[pageSize - 1]);
        Assert.Equal(0.3f, arr[pageSize]);
        Assert.Equal(0.4f, arr[pageSize + 9]);
    }

    [Fact]
    public void Indexer_OutOfRange_Throws()
    {
        using var arr = new BigArray<float>(5);
        Assert.Throws<IndexOutOfRangeException>(() => arr[-1]);
        Assert.Throws<IndexOutOfRangeException>(() => arr[5]);
    }

    [Fact]
    public void BlockCount_CorrectForPartialPage()
    {
        int pageSize = 1024 * 1024;
        using var arr = new BigArray<float>(pageSize + 1);
        Assert.Equal(2, arr.BlockCount);
    }

    [Fact]
    public void GetBlock_ReturnsCorrectSpan()
    {
        int pageSize = 1024 * 1024;
        using var arr = new BigArray<float>(pageSize + 10);
        arr[pageSize + 5] = 42.0f;

        var span = arr.GetBlock(1, out int offset, out int count);
        Assert.Equal(42.0f, span[5]);
        Assert.Equal(pageSize + 10 - pageSize, count); // last page partial
    }

    [Fact]
    public void ZeroLength_CreatesSuccessfully()
    {
        using var arr = new BigArray<float>(0);
        Assert.Equal(0L, arr.Length);
        Assert.Equal(0, arr.BlockCount);
    }

    [Fact]
    public void NegativeLength_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new BigArray<float>(-1));
    }

    [Fact]
    public void SpanBlockIndex_OutOfRange_Throws()
    {
        using var arr = new BigArray<float>(10);
        Assert.Throws<ArgumentOutOfRangeException>(() => arr.GetBlock(1, out _, out _));
    }
}
