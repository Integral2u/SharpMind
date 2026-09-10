using SharpMind.Model;
using SharpMind.Core.Tensors;

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

        [Fact]
        public void QuantizedKVCache_TrimToLast_RetainsLatestRows()
        {
            using var cache = new QuantizedKVCache(1, 1, 8, 32);
            using var keys = new Tensor<float>(1, 4, 1, 32);
            using var values = new Tensor<float>(1, 4, 1, 32);
            for (int s = 0; s < 4; s++)
            {
                for (int i = 0; i < 32; i++)
                {
                    keys.Data[s * 32 + i] = s * 100 + i;
                    values.Data[s * 32 + i] = -(s * 100 + i);
                }
            }
            cache.Update(keys, values, 1, 32);
            cache.TrimToLast(3);

            using var expected = new QuantizedKVCache(1, 1, 8, 32);
            using var expectedKeys = new Tensor<float>(1, 3, 1, 32);
            using var expectedValues = new Tensor<float>(1, 3, 1, 32);
            keys.Data.Slice(32, 96).CopyTo(expectedKeys.Data);
            values.Data.Slice(32, 96).CopyTo(expectedValues.Data);
            expected.Update(expectedKeys, expectedValues, 1, 32);

            var (_, actualKeys, actualValues) = ((int, byte[], byte[]))cache.Snapshot()!;
            var (_, expectedKeyBytes, expectedValueBytes) = ((int, byte[], byte[]))expected.Snapshot()!;
            Assert.Equal(expectedKeyBytes, actualKeys);
            Assert.Equal(expectedValueBytes, actualValues);
        }

        [Fact]
        public void KVCache_SnapshotRoundTrip_PreservesEveryHead()
        {
            using var cache = new KVCache(1, 2, 8, 2);
            using var keys = new Tensor<float>(1, 3, 2, 2);
            using var values = new Tensor<float>(1, 3, 2, 2);
            for (int i = 0; i < keys.ElementCount; i++)
            {
                keys.Data[i] = i + 1;
                values.Data[i] = -i - 1;
            }
            cache.Update(keys, values, 2, 2);
            var snapshot = cache.Snapshot();

            using var restored = new KVCache(1, 2, 8, 2);
            restored.Restore(snapshot);
            var (expectedPosition, expectedKeys, expectedValues) = ((int, float[], float[]))snapshot!;
            var (actualPosition, actualKeys, actualValues) = ((int, float[], float[]))restored.Snapshot()!;
            Assert.Equal(expectedPosition, actualPosition);
            Assert.Equal(expectedKeys, actualKeys);
            Assert.Equal(expectedValues, actualValues);
        }

        [Fact]
        public void QuantizedKVCache_SnapshotRoundTrip_PreservesEveryHead()
        {
            using var cache = new QuantizedKVCache(1, 2, 8, 32);
            using var keys = new Tensor<float>(1, 3, 2, 32);
            using var values = new Tensor<float>(1, 3, 2, 32);
            for (int i = 0; i < keys.ElementCount; i++)
            {
                keys.Data[i] = i - 50;
                values.Data[i] = 50 - i;
            }
            cache.Update(keys, values, 2, 32);
            var snapshot = cache.Snapshot();

            using var restored = new QuantizedKVCache(1, 2, 8, 32);
            restored.Restore(snapshot);
            var (expectedPosition, expectedKeys, expectedValues) = ((int, byte[], byte[]))snapshot!;
            var (actualPosition, actualKeys, actualValues) = ((int, byte[], byte[]))restored.Snapshot()!;
            Assert.Equal(expectedPosition, actualPosition);
            Assert.Equal(expectedKeys, actualKeys);
            Assert.Equal(expectedValues, actualValues);
        }
    }
}
