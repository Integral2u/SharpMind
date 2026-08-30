using SharpMind.Core.Memory;

namespace SharpMind.Tests.Memory;

/// <summary>Keeps pool state deterministic for the correctness probe.</summary>
[CollectionDefinition("NativeBufferPool (serialized)", DisableParallelization = true)]
public sealed class NativeBufferPoolCollection;

/// <summary>
/// Regression test for <see cref="NativeBufferPool{T}"/>: <c>Return</c> used to
/// increment <c>bucket.Count</c> twice per successful pool (once as the slot
/// reservation, once again on push), while <c>Rent</c> decrements exactly once.
/// Each return+rent cycle therefore netted <c>Count += 1</c> beyond the real
/// stack depth, so <c>reservedCount &lt;= MaxBuffersPerBucket</c> failed at
/// roughly half the configured capacity and the pool silently fell back to
/// free/re-allocate for <em>all</em> subsequent buffers — defeating pooling
/// under sustained training load.
///
/// A tightly coupled index over <c>Count</c> (bucket.Capacity) is exactly the
/// bug: over many rent/return cycles with a single reusable buffer, a healthy
/// pool must keep handing out the same <see cref="NativeBuffer{T}"/> instance.
/// Once <c>Count</c> drifts past the cap the buffer gets freed and every later
/// rent allocates a brand-new instance.
///
/// The pool is generic over <c>T</c> and stateless except for per-<c>T</c>
/// static buckets, and the product only ever allocates buffers for
/// <c>float</c> and <c>int</c>. Probing with <c>double</c> therefore hits a
/// bucket (1,048,576 elements) that no other test — running in any of the
/// parallel worker threads — can ever touch, so the reuse assertion below is
/// deterministic rather than hostage to what the rest of the suite is renting
/// at the same instant.
/// </summary>
[Collection("NativeBufferPool (serialized)")]
public sealed class NativeBufferPoolTests
{
    // GetBucket -> 1,048,576; inside the poolable <= 1 Mi-element gate, but used
    // only by the double pool (see above), so it stays private to this test.
    private const int BucketElements = 1_000_003;

    [Fact]
    public void Rent_Reuses_SameInstance_WellBeyond_BucketCapacity()
    {
        const int cycles = 500;

        // This tests the pool's index bookkeeping (the bucket.Count double-increment bug), not
        // its free-headroom heuristic. The MaxTotalMemoryMB pressure valve is a global counter
        // across every pooled type, and earlier behaviour treated "reuse stops" as proof of the
        // bug — but once the GPU tests were merged in-process their (much larger) cumulative
        // native-float allocations push TotalMemoryUsed toward/over the 512 MB cap, which makes
        // Return() free this ~8 MB double buffer instead of pooling it. That is the cap working,
        // not the bug. Pin the cap high for the duration so reuse is decided purely by Count.
        long savedCap = NativeBufferPoolConfig.MaxTotalMemoryMB;
        NativeBufferPoolConfig.MaxTotalMemoryMB = long.MaxValue / (1024L * 1024L);

        NativeBufferPool<double>.Clear();
        try
        {
            NativeBuffer<double>? reused = null;
            using (var first = NativeBufferPool<double>.Rent(BucketElements))
            {
                reused = first;
            }

            for (int i = 0; i < cycles; i++)
            {
                using var b = NativeBufferPool<double>.Rent(BucketElements);
                Assert.True(ReferenceEquals(b, reused),
                    $"Pool stopped reusing at cycle {i}: new instance allocated — " +
                    "bucket.Count drifted past the cap (double-increment bug).");
            }
        }
        finally
        {
            NativeBufferPool<double>.Clear();
            NativeBufferPoolConfig.MaxTotalMemoryMB = savedCap;
        }
    }
}