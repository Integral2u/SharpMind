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
/// </summary>
[Collection("NativeBufferPool (serialized)")]
public sealed class NativeBufferPoolTests
{
    private const int BucketElements = 1_000_003; // GetBucket -> 1,048,576; within the poolable <= 1 Mi-element gate

    [Fact]
    public void Rent_Reuses_SameInstance_WellBeyond_BucketCapacity()
    {
        const int maxPerBucket = 4;
        const long maxMemMb = 4_096; // far above suite peak so the memory gate never trips
        const int cycles = 500;

        int oldMax = NativeBufferPoolConfig.MaxBuffersPerBucket;
        long oldMemMb = NativeBufferPoolConfig.MaxTotalMemoryMB;
        NativeBufferPool<float>.Clear();
        try
        {
            NativeBufferPoolConfig.MaxBuffersPerBucket = maxPerBucket;
            NativeBufferPoolConfig.MaxTotalMemoryMB = maxMemMb;

            NativeBuffer<float>? reused = null;
            using (var first = NativeBufferPool<float>.Rent(BucketElements))
            {
                reused = first;
            }

            for (int i = 0; i < cycles; i++)
            {
                using var b = NativeBufferPool<float>.Rent(BucketElements);
                Assert.True(ReferenceEquals(b, reused),
                    $"Pool stopped reusing at cycle {i}: new instance allocated — " +
                    "bucket.Count drifted past the cap (double-increment bug).");
            }
        }
        finally
        {
            NativeBufferPoolConfig.MaxBuffersPerBucket = oldMax;
            NativeBufferPoolConfig.MaxTotalMemoryMB = oldMemMb;
            NativeBufferPool<float>.Clear();
        }
    }
}