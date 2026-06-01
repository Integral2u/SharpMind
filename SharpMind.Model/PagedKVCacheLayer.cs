using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model;

/// <summary>
/// Wrapper that provides a single-layer KV Cache interface using paged memory internally.
/// This is a drop-in replacement for KVCache that uses PagedAttention internally.
/// </summary>
public sealed class PagedKVCacheLayer(int batchSize, int numKvHeads, int maxSeqLen, int headDim, int pageSize = PagedKVCache.DefaultPageSize) : IKVCache
{
    private readonly PagedKVCache _cache = new(batchSize, numKvHeads, maxSeqLen, headDim, pageSize);
    private readonly int _maxSeqLen = maxSeqLen;
    private int _currentPosition = 0;

    public int Length => _currentPosition;
    public int MaxSeqLen => _maxSeqLen;
    public bool IsFull => _currentPosition >= _maxSeqLen;

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        _cache.Update(k, v, numKvHeads, headDim);
        _currentPosition += k.Shape[1];
    }

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead)
        => _cache.GetKeyPtr(batchIdx, position, kvHead);

    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead)
        => _cache.GetValuePtr(batchIdx, position, kvHead);

    public void Reset()
    {
        _cache.Reset();
        _currentPosition = 0;
    }

    public void TrimToLast(int keepTokens)
    {
        _cache.TrimToLast(keepTokens);
        _currentPosition = Math.Min(_currentPosition, keepTokens);
    }

    public void Dispose() => _cache.Dispose();
}