using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model;

/// <summary>
/// Wrapper that provides a single-layer KV Cache interface using paged memory internally.
/// This is a drop-in replacement for KVCache that uses PagedAttention internally.
/// </summary>
public sealed class PagedKVCacheLayer : IDisposable
{
    private readonly PagedKVCache _cache;
    private readonly int _maxSeqLen;
    private int _currentPosition;

    public PagedKVCacheLayer(int batchSize, int numKvHeads, int maxSeqLen, int headDim, int pageSize = PagedKVCache.DefaultPageSize)
    {
        _cache = new PagedKVCache(batchSize, numKvHeads, maxSeqLen, headDim, pageSize);
        _maxSeqLen = maxSeqLen;
        _currentPosition = 0;
    }

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

/// <summary>
/// Factory for creating paged KV cache arrays for all transformer layers.
/// </summary>
public static class PagedKVCacheFactory
{
    public static PagedKVCacheLayer[] CreateArray(int numLayers, int batchSize, int numKvHeads, int maxSeqLen, int headDim, int pageSize = PagedKVCache.DefaultPageSize)
    {
        var caches = new PagedKVCacheLayer[numLayers];
        for (int i = 0; i < numLayers; i++)
            caches[i] = new PagedKVCacheLayer(batchSize, numKvHeads, maxSeqLen, headDim, pageSize);
        return caches;
    }

    public static PagedKVCacheLayer[] CreateArray(ModelConfig config, int batchSize = 1)
    {
        return CreateArray(
            config.NumLayers,
            batchSize,
            config.NumKvHeads,
            config.MaxSeqLen,
            config.HeadDim);
    }
}