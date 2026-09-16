using SharpMind.Core.Embeddings;

namespace SharpMind.Tests.Core;

// The table cache is process-wide: a test building RoPE tables in parallel could evict an entry
// this test expects to still be cached.
[Collection("Non-Parallel")]
public sealed class RoPETableCacheTests
{
    [Fact]
    public void TableCache_KeepsAFixedNumberOfConfigurationsAndEvictsTheLeastRecentlyUsed()
    {
        // Reloading a model shares its rotary tables, but a process that loads many configurations
        // must not keep every table it ever built.
        static RoPE Build(int config) => new(headDim: 8, maxSeqLen: 4, theta: 20_000f + config);
        int capacity = RoPE.TableCacheCapacity;

        // Fill the cache with our own configurations, pushing out any other test's tables.
        var built = Enumerable.Range(0, capacity).Select(Build).ToArray();
        Assert.True(built[0].CosTable == Build(0).CosTable, "a cached configuration should share its tables");

        // Configuration 0 was just used, so one more configuration evicts configuration 1.
        Build(capacity);
        Assert.True(built[0].CosTable == Build(0).CosTable, "the most recently used configuration should still be cached");
        for (int config = 2; config < capacity; config++)
            Assert.True(built[config].CosTable == Build(config).CosTable, $"configuration {config} should still be cached");
        Assert.False(built[1].CosTable == Build(1).CosTable, "the least recently used configuration should have been evicted");
    }
}
