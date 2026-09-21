using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Tests.Model;

/// <summary>
/// <see cref="Int8KVCache"/> stores every K/V row of every kv head as int8 (symmetric,
/// scale = max|x| / 127): a value row is one float scale followed by headDim values, a key row is
/// one float scale per 16 values, all scales first, followed by headDim values. That is the layout
/// the int8 attention kernel reads. Snapshots are those bytes per head, positions in order.
/// </summary>
public sealed class Int8KVCacheTests
{
    private static Tensor<float> Rows(int seqLen, int numKvHeads, int headDim, Func<int, float> value)
    {
        var t = new Tensor<float>(1, seqLen, numKvHeads, headDim);
        for (int i = 0; i < t.ElementCount; i++) t.Data[i] = value(i);
        return t;
    }

    private static (int Position, byte[] Keys, byte[] Values) SnapshotOf(IKVCache cache) =>
        ((int, byte[], byte[]))cache.Snapshot()!;

    // Spelled out here rather than read from Int8KVCache, so the cache cannot drift unnoticed.
    private const int KeyBlock = 16;
    private static int KeyRowBytes(int headDim) => (headDim + KeyBlock - 1) / KeyBlock * sizeof(float) + headDim;

    /// <summary>Checks one quantized value row against the float row it came from.</summary>
    private static void AssertRow(byte[] bytes, int offset, ReadOnlySpan<float> source) =>
        AssertBlock(bytes, offset, offset + sizeof(float), source);

    /// <summary>Checks one quantized key row, block by block, against the float row it came from.</summary>
    private static void AssertKeyRow(byte[] bytes, int offset, ReadOnlySpan<float> source)
    {
        int blocks = (source.Length + KeyBlock - 1) / KeyBlock;
        for (int b = 0; b < blocks; b++)
        {
            int start = b * KeyBlock, length = Math.Min(KeyBlock, source.Length - start);
            AssertBlock(bytes, offset + b * sizeof(float), offset + blocks * sizeof(float) + start, source.Slice(start, length));
        }
    }

    private static void AssertBlock(byte[] bytes, int scaleOffset, int dataOffset, ReadOnlySpan<float> source)
    {
        float amax = 0f;
        foreach (float x in source) amax = MathF.Max(amax, MathF.Abs(x));
        float scale = BitConverter.ToSingle(bytes, scaleOffset);
        Assert.Equal(amax / 127f, scale, 6);
        for (int d = 0; d < source.Length; d++)
        {
            sbyte q = (sbyte)bytes[dataOffset + d];
            Assert.InRange(q, (sbyte)-127, (sbyte)127);
            Assert.True(MathF.Abs(q * scale - source[d]) <= scale / 2 + 1e-6f,
                $"element {d}: {q} * {scale} is not the nearest step to {source[d]}");
        }
    }

    [Fact]
    public void DescribesItselfAsAContiguousInt8Cache()
    {
        using var cache = new Int8KVCache(1, 2, 8, 4);
        Assert.True(cache.IsQuantized);
        Assert.True(cache.IsContiguous);
        Assert.Equal(QuantDType.I8, cache.QuantKind);
        Assert.Equal(8, cache.MaxSeqLen);
        Assert.Equal(0, cache.Length);
    }

    [Fact]
    public void Update_StoresValueRowsWithOneScaleAndKeyRowsWithOnePerBlock()
    {
        // headDim 20: a key row is a whole block of 16 and a short one of 4.
        const int seqLen = 3, heads = 2, headDim = 20;
        using var cache = new Int8KVCache(1, heads, 8, headDim);
        // Row (s=1, h=0) starts at 40. Keys: its first block is all zeros inside a non-zero row.
        // Values: the whole row is zeros. The others mix signs and magnitudes.
        using var keys = Rows(seqLen, heads, headDim, i => i is >= 40 and < 56 ? 0f : (i % 5 - 2) * 1.37f + i * 0.01f);
        using var values = Rows(seqLen, heads, headDim, i => i is >= 40 and < 60 ? 0f : -(i % 3) * 250.5f + 0.25f);
        cache.Update(keys, values, heads, headDim);

        var (position, k, v) = SnapshotOf(cache);
        Assert.Equal(seqLen, position);
        int keyRowBytes = KeyRowBytes(headDim), valueRowBytes = sizeof(float) + headDim;
        Assert.Equal(28, keyRowBytes);
        Assert.Equal(heads * seqLen * keyRowBytes, k.Length);
        Assert.Equal(heads * seqLen * valueRowBytes, v.Length);
        for (int h = 0; h < heads; h++)
            for (int s = 0; s < seqLen; s++)
            {
                int source = (s * heads + h) * headDim;
                AssertKeyRow(k, (h * seqLen + s) * keyRowBytes, keys.Data.Slice(source, headDim));
                AssertRow(v, (h * seqLen + s) * valueRowBytes, values.Data.Slice(source, headDim));
            }
    }

    /// <summary>
    /// What the key blocks are for: an outlier channel takes the int8 range of its own block only.
    /// With one scale per row the step would be 1000 / 127 and every small value would round to zero.
    /// </summary>
    [Fact]
    public void KeyOutlierChannel_DoesNotFlattenTheOtherBlocks()
    {
        const int headDim = 32;
        using var cache = new Int8KVCache(1, 1, 4, headDim);
        using var keys = Rows(1, 1, headDim, i => i == 3 ? 1000f : 0.5f - i * 0.01f);
        cache.Update(keys, keys, 1, headDim);

        var k = SnapshotOf(cache).Keys;
        float secondScale = BitConverter.ToSingle(k, sizeof(float));
        for (int d = KeyBlock; d < headDim; d++)
        {
            float restored = (sbyte)k[2 * sizeof(float) + d] * secondScale;
            Assert.True(MathF.Abs(restored - keys.Data[d]) <= 0.002f, $"element {d}: {restored} vs {keys.Data[d]}");
        }
    }

    [Fact]
    public void TrimToLast_RetainsTheLatestRows()
    {
        using var cache = new Int8KVCache(1, 1, 8, 32);
        using var keys = Rows(4, 1, 32, i => i % 32 + (i / 32) * 100);
        using var values = Rows(4, 1, 32, i => -(i % 32 + (i / 32) * 100));
        cache.Update(keys, values, 1, 32);
        cache.TrimToLast(3);

        using var expected = new Int8KVCache(1, 1, 8, 32);
        using var expectedKeys = Rows(3, 1, 32, i => keys.Data[32 + i]);
        using var expectedValues = Rows(3, 1, 32, i => values.Data[32 + i]);
        expected.Update(expectedKeys, expectedValues, 1, 32);

        Assert.Equal(3, cache.Length);
        Assert.Equal(SnapshotOf(expected).Keys, SnapshotOf(cache).Keys);
        Assert.Equal(SnapshotOf(expected).Values, SnapshotOf(cache).Values);
    }

    [Fact]
    public void Truncate_Reset_AndCapacity()
    {
        using var cache = new Int8KVCache(1, 1, 4, 8);
        using var rows = Rows(3, 1, 8, i => i + 1);
        cache.Update(rows, rows, 1, 8);
        cache.Truncate(2);
        Assert.Equal(2, cache.Length);
        cache.Truncate(5);
        Assert.Equal(2, cache.Length);

        using var two = Rows(2, 1, 8, i => -i);
        cache.Update(two, two, 1, 8);
        Assert.True(cache.IsFull);
        Assert.Throws<InvalidOperationException>(() => cache.Update(two, two, 1, 8));

        cache.Reset();
        Assert.Equal(0, cache.Length);
        Assert.Null(cache.Snapshot());
    }

    [Fact]
    public void SnapshotBytes_RoundTripsEveryHead()
    {
        using var cache = new Int8KVCache(1, 2, 8, 32);
        using var keys = Rows(3, 2, 32, i => i - 50);
        using var values = Rows(3, 2, 32, i => 50 - i * 0.5f);
        cache.Update(keys, values, 2, 32);
        byte[]? bytes = ((IKVCache)cache).SnapshotBytes();

        using var restored = new Int8KVCache(1, 2, 8, 32);
        ((IKVCache)restored).RestoreBytes(bytes);
        Assert.Equal(3, restored.Length);
        Assert.Equal(SnapshotOf(cache).Keys, SnapshotOf(restored).Keys);
        Assert.Equal(SnapshotOf(cache).Values, SnapshotOf(restored).Values);
    }

    /// <summary>
    /// At headDim 64 a Q8_0 row (2 blocks x 34 bytes) and an int8 value row (4 + 64 bytes) are the
    /// same size, so a snapshot tagged only by its tuple type could restore values into the other
    /// cache as garbage. The formats must refuse each other.
    /// </summary>
    [Fact]
    public void RestoreBytes_RejectsAQ8_0SnapshotOfTheSameSize_AndViceVersa()
    {
        using var rows = Rows(2, 1, 64, i => i * 0.1f - 3f);

        using var q8 = new QuantizedKVCache(1, 1, 8, 64);
        q8.Update(rows, rows, 1, 64);
        byte[]? q8Bytes = ((IKVCache)q8).SnapshotBytes();
        using var int8 = new Int8KVCache(1, 1, 8, 64);
        Assert.Throws<InvalidDataException>(() => ((IKVCache)int8).RestoreBytes(q8Bytes));

        int8.Update(rows, rows, 1, 64);
        byte[]? int8Bytes = ((IKVCache)int8).SnapshotBytes();
        using var otherQ8 = new QuantizedKVCache(1, 1, 8, 64);
        Assert.Throws<InvalidDataException>(() => ((IKVCache)otherQ8).RestoreBytes(int8Bytes));
    }
}
