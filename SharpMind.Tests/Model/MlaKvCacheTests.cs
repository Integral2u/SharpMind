using System;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Core.Tensors;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// The MLA cache holds one <c>kv_lora_rank + rope</c> latent per position. The
/// interesting properties are the ones every other cache gets for free from
/// storing K and V separately: it must append in order, truncate, keep the last
/// n rows, and round-trip through the tag-2 snapshot the session writer uses.
/// </summary>
public sealed class MlaKvCacheTests
{
    private const int Latent = 8;

    private static Tensor<float> Rows(int seqLen, float start)
    {
        var t = new Tensor<float>(1, seqLen, Latent);
        for (int s = 0; s < seqLen; s++)
            for (int j = 0; j < Latent; j++)
                t.Data[s * Latent + j] = start + s * 100 + j;
        return t;
    }

    private static unsafe float At(IKVCache cache, int position, int j)
        => cache.GetKeyPtr(0, position, 0)[j];

    [Fact]
    public void Update_AppendsLatents_InOrder()
    {
        using var cache = new MlaKvCache(1, 16, Latent);
        using var first = Rows(2, 0f);
        cache.Update(first, first, 1, Latent);

        Assert.Equal(2, cache.Length);
        Assert.Equal(0f, At(cache, 0, 0));
        Assert.Equal(101f, At(cache, 1, 1));

        using var second = Rows(1, 500f);
        cache.Update(second, second, 1, Latent);
        Assert.Equal(3, cache.Length);
        Assert.Equal(503f, At(cache, 2, 3));
    }

    [Fact]
    public void Update_RejectsAWrongWidth()
    {
        using IKVCache cache = new MlaKvCache(1, 4, Latent);
        using var rows = Rows(1, 0f);
        Assert.Throws<InvalidOperationException>(() => cache.Update(rows, rows, 2, Latent));
        Assert.Throws<InvalidOperationException>(() => cache.Update(rows, rows, 1, Latent - 1));
    }

    [Fact]
    public void Truncate_And_TrimToLast_MoveTheRows()
    {
        using var cache = new MlaKvCache(1, 16, Latent);
        using var rows = Rows(4, 0f);
        cache.Update(rows, rows, 1, Latent);

        cache.TrimToLast(2);
        Assert.Equal(2, cache.Length);
        Assert.Equal(200f, At(cache, 0, 0));   // rows 2 and 3 moved to the front
        Assert.Equal(301f, At(cache, 1, 1));

        cache.Truncate(1);
        Assert.Equal(1, cache.Length);

        cache.Reset();
        Assert.Equal(0, cache.Length);
    }

    [Fact]
    public void Snapshot_RoundTripsThroughTheTag2Format()
    {
        using IKVCache source = new MlaKvCache(1, 16, Latent);
        using var rows = Rows(3, 10f);
        source.Update(rows, rows, 1, Latent);

        byte[]? bytes = source.SnapshotBytes();
        Assert.NotNull(bytes);

        using IKVCache restored = new MlaKvCache(1, 16, Latent);
        restored.RestoreBytes(bytes);
        Assert.Equal(3, restored.Length);
        for (int s = 0; s < 3; s++)
            for (int j = 0; j < Latent; j++)
                Assert.Equal(At(source, s, j), At(restored, s, j));
    }

    [Fact]
    public void Snapshot_OfAnEmptyCacheIsNull()
    {
        using IKVCache cache = new MlaKvCache(1, 4, Latent);
        Assert.Null(cache.Snapshot());
        Assert.Null(cache.SnapshotBytes());
    }

    [Fact]
    public void Builder_ProducesAnMlaCache()
    {
        // The generic builder must hand MLA models the latent cache; the default
        // interface method keeps every other builder on the expanded one.
        IKVCacheBuilder builder = new KVCacherBuilder();
        using var cache = builder.CreateMlaKvCache(1, 8, 576);
        Assert.IsType<MlaKvCache>(cache);
        Assert.Equal(576, ((MlaKvCache)cache).LatentDim);
    }
}
