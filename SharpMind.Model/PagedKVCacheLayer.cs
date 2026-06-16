using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Wrapper that provides a single-layer KV Cache interface using paged memory internally.
/// This is a drop-in replacement for KVCache that uses PagedAttention internally.
/// </summary>
public sealed class PagedKVCacheLayer(int batchSize, int numKvHeads, int maxSeqLen, int headDim, int pageSize = PagedKVCache.DefaultPageSize) : IKVCache
{
    private readonly PagedKVCache _cache = new(batchSize, numKvHeads, maxSeqLen, headDim, pageSize);
    private readonly int _maxSeqLen = maxSeqLen;

    public int Length => _cache.Length;
    public int MaxSeqLen => _maxSeqLen;
    public bool IsFull => _cache.IsFull;
    public bool IsContiguous => false;

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim) => _cache.Update(k, v, numKvHeads, headDim);

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead)
    {
        return _cache.GetKeyPtr(batchIdx, position, kvHead);
    }

    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead) => _cache.GetValuePtr(batchIdx, position, kvHead);

    public void Reset() => _cache.Reset();

    public void TrimToLast(int keepTokens) => _cache.TrimToLast(keepTokens);

    public void Truncate(int length) => _cache.Truncate(length);

    public void Dispose() => _cache.Dispose();
    public object? Snapshot() => _cache.Snapshot();
    public void Restore(object? snapshot) { _cache.Restore(snapshot); }
}
