using SharpMind.Model;

namespace SharpMind.Tests.Model
{
    public sealed class KVCacheCeilingTests
    {
        [Fact]
        public void KVCache_CapacityOverflow_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new KVCache(1, 1, int.MaxValue, 2));
        }

        [Fact]
        public void PagedKVCache_StrideOverflow_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new PagedKVCache(1, 1, 1024, int.MaxValue, pageSize: 2));
        }

        [Fact]
        public void QuantizedKVCache_HeadStrideOverflow_Throws()
        {
            Assert.Throws<ArgumentOutOfRangeException>(
                () => new QuantizedKVCache(1, 1, 1 << 30, 34));
        }
    }
}