using SharpMind.Core.Memory;
using Xunit;

namespace SharpMind.Tests.Memory;

public class ByteBufferPoolTests
{
    [Fact]
    public void Acquire_AllocatesFresh_UntilReleased_ThenReuses()
    {
        var pool = new ByteBufferPool();
        var a = pool.Acquire(1024);
        var fresh = pool.Acquire(1024);
        Assert.Equal(1024, a.Length);
        Assert.NotSame(a, fresh);

        pool.Release(a);
        var b = pool.Acquire(1024);
        Assert.Same(a, b); // reuse — no allocation
    }

    [Fact]
    public void Acquire_DistinctSizes_DoNotInterfere()
    {
        var pool = new ByteBufferPool();
        var big = pool.Acquire(65536);
        var small = pool.Acquire(16);
        pool.Release(big);
        pool.Release(small);

        Assert.Same(small, pool.Acquire(16));
        Assert.Same(big, pool.Acquire(65536));
    }

    [Fact]
    public void Acquire_Negative_Throws()
    {
        var pool = new ByteBufferPool();
        Assert.Throws<ArgumentOutOfRangeException>(() => pool.Acquire(-1));
    }

    [Fact]
    public void Release_Null_IsNoOp()
    {
        var pool = new ByteBufferPool();
        pool.Release(null);
        Assert.Equal(0, pool.RetainedCount);
        Assert.Equal(0, pool.RetainedBytes);
    }

    [Fact]
    public void ExplicitBudget_EvictsOldest()
    {
        var pool = new ByteBufferPool { MaxRetainedBytes = 100 };
        pool.Release(pool.Acquire(64));
        Assert.Equal(1, pool.RetainedCount);

        pool.Release(pool.Acquire(64)); // 128 > 100 → evicts one
        Assert.Equal(1, pool.RetainedCount);
        Assert.Equal(64, pool.RetainedBytes);
    }

    [Fact]
    public void WindowBudget_ScalesWithLargestTensorAndResidentLayers()
    {
        var pool = new ByteBufferPool { ResidentLayers = 2 };
        var tenors = new[] { pool.Acquire(1000), pool.Acquire(1000) }; // largest seen = 1000
        pool.Release(tenors[0]);
        pool.Release(tenors[1]);
        // cap = 1000 * (2+2) = 4000 → both retained
        Assert.Equal(2, pool.RetainedCount);

        var extra = pool.Acquire(1000); // still 1000 → cap unchanged
        pool.Release(extra);
        Assert.Equal(3, pool.RetainedCount);

        // A larger tensor raises the cap; total must never exceed largest * (residents + 2).
        var big = pool.Acquire(2000);
        pool.Release(big);
        Assert.True(pool.RetainedBytes <= 2000 * (2 + 2));
    }

    [Fact]
    public void ConcurrentAcquireRelease_IsSafe()
    {
        var pool = new ByteBufferPool();
        int workers = 8, rounds = 2000;
        var ready = new ManualResetEventSlim(false);
        var t = new Task[workers];
        for (int w = 0; w < workers; w++)
        {
            t[w] = Task.Run(() =>
            {
                ready.Wait();
                for (int i = 0; i < rounds; i++)
                {
                    var b = pool.Acquire(256 + (i % 4) * 64);
                    b[0] = 1;
                    pool.Release(b);
                }
            });
        }
        ready.Set();
        Task.WaitAll(t);
        Assert.True(pool.RetainedCount <= workers); // bounded by per-size reuse, no loss
    }
}