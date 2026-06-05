using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Paged KV Cache - uses fixed-size pages instead of contiguous allocation.
/// This reduces memory fragmentation and enables efficient sliding windows.
/// 
/// Layout: [Batch, NumKvHeads, NumPages, PageSize, HeadDim]
/// Each page is PageSize tokens. Pages are allocated on-demand.
/// </summary>
public sealed class PagedKVCache : IDisposable
{
    public const int DefaultPageSize = 32;

    private readonly int _batchSize;
    private readonly int _numKvHeads;
    private readonly int _maxSeqLen;
    private readonly int _headDim;
    private readonly int _pageSize;

    private readonly Tensor<float>[,] _keysPages;   // [numKvHeads, numPages]
    private readonly Tensor<float>[,] _valuePages;
    private readonly int _numPages;
    private readonly bool[] _pageAllocated;  // per batch

    private int _currentPosition;
    private bool _disposed;

    public PagedKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim, int pageSize = DefaultPageSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numKvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        _batchSize = batchSize;
        _numKvHeads = numKvHeads;
        _maxSeqLen = maxSeqLen;
        _headDim = headDim;
        _pageSize = pageSize;

        _numPages = (maxSeqLen + pageSize - 1) / pageSize;
        
        _keysPages = new Tensor<float>[_numKvHeads, _numPages];
        _valuePages = new Tensor<float>[_numKvHeads, _numPages];
        _pageAllocated = new bool[batchSize * _numPages];

        for (int p = 0; p < _numPages; p++)
        {
            for (int h = 0; h < _numKvHeads; h++)
            {
                _keysPages[h, p] = new Tensor<float>(batchSize, pageSize, headDim);
                _valuePages[h, p] = new Tensor<float>(batchSize, pageSize, headDim);
            }
        }
    }

    public int Length => _currentPosition;
    public int PageSize => _pageSize;
    public int MaxSeqLen => _maxSeqLen;
    public bool IsFull => _currentPosition >= _maxSeqLen;

    /// <summary>
    /// Returns the page index and offset within page for a given position.
    /// </summary>
    public (int pageIdx, int offset) GetPageInfo(int position)
    {
        int pageIdx = position / _pageSize;
        int offset = position % _pageSize;
        return (pageIdx, offset);
    }

    /// <summary>
    /// Gets a pointer to the Key values at the given position.
    /// </summary>
    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead)
    {
        var (pageIdx, offset) = GetPageInfo(position);
        var tensor = _keysPages[kvHead, pageIdx];
        return tensor.DataPtr + (long)batchIdx * (_pageSize * _headDim) + offset * _headDim;
    }

    /// <summary>
    /// Gets a pointer to the Value values at the given position.
    /// </summary>
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead)
    {
        var (pageIdx, offset) = GetPageInfo(position);
        var tensor = _valuePages[kvHead, pageIdx];
        return tensor.DataPtr + (long)batchIdx * (_pageSize * _headDim) + offset * _headDim;
    }

    /// <summary>
    /// Copies K/V data into the cache at the current position.
    /// </summary>
    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        ThrowIfDisposed();

        int batch = k.Shape[0];
        int seqLen = k.Shape[1];

        if (_currentPosition + seqLen > _maxSeqLen)
            throw new InvalidOperationException(
                $"KV cache overflow: {_currentPosition} + {seqLen} > {_maxSeqLen}");

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int dstPos = _currentPosition + s;
                var (pageIdx, offset) = GetPageInfo(dstPos);

                for (int h = 0; h < numKvHeads; h++)
                {
                    unsafe
                    {
                        float* srcK = k.DataPtr + (long)b * seqLen * numKvHeads * headDim
                                        + (long)s * numKvHeads * headDim
                                        + (long)h * headDim;
                        float* dstK = _keysPages[h, pageIdx].DataPtr + (long)b * _pageSize * _headDim
                                                    + offset * _headDim;

                        for (int d = 0; d < headDim; d++)
                            dstK[d] = srcK[d];

                        float* srcV = v.DataPtr + (long)b * seqLen * numKvHeads * headDim
                                        + (long)s * numKvHeads * headDim
                                        + (long)h * headDim;
                        float* dstV = _valuePages[h, pageIdx].DataPtr + (long)b * _pageSize * _headDim
                                                              + offset * _headDim;

                        for (int d = 0; d < headDim; d++)
                            dstV[d] = srcV[d];
                    }
                }
            }
        }

        _currentPosition += seqLen;
    }

    /// <summary>
    /// Resets cache to empty without deallocating pages.
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();
        _currentPosition = 0;
    }

    /// <summary>
    /// Trims to keep only the last N pages (efficient sliding window).
    /// </summary>
    public void TrimToLast(int keepTokens)
    {
        ThrowIfDisposed();
        if (keepTokens >= _currentPosition) return;
        
        _currentPosition = keepTokens;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        for (int p = 0; p < _numPages; p++)
        {
            for (int h = 0; h < _numKvHeads; h++)
            {
                _keysPages[h, p]?.Dispose();
                _valuePages[h, p]?.Dispose();
            }
        }
    }

    public object? Snapshot()
    {
        ThrowIfDisposed();
        if (_currentPosition == 0) return null;
        int total = 0;
        for (int h = 0; h < _numKvHeads; h++)
            for (int p = 0; p < _numPages; p++)
                total += _keysPages[h, p].ElementCount * 2;
        var data = new float[total];
        int idx = 0;
        for (int h = 0; h < _numKvHeads; h++)
            for (int p = 0; p < _numPages; p++)
            {
                var src = _keysPages[h, p].Data;
                src.CopyTo(data.AsSpan(idx, src.Length));
                idx += src.Length;
            }
        for (int h = 0; h < _numKvHeads; h++)
            for (int p = 0; p < _numPages; p++)
            {
                var src = _valuePages[h, p].Data;
                src.CopyTo(data.AsSpan(idx, src.Length));
                idx += src.Length;
            }
        return (_currentPosition, data);
    }

    public void Restore(object? snapshot)
    {
        ThrowIfDisposed();
        if (snapshot is null) return;
        var (pos, data) = ((int, float[]))snapshot;
        _currentPosition = pos;
        int idx = 0;
        for (int h = 0; h < _numKvHeads; h++)
            for (int p = 0; p < _numPages; p++)
            {
                var span = _keysPages[h, p].Data;
                data.AsSpan(idx, span.Length).CopyTo(span);
                idx += span.Length;
            }
        for (int h = 0; h < _numKvHeads; h++)
            for (int p = 0; p < _numPages; p++)
            {
                var span = _valuePages[h, p].Data;
                data.AsSpan(idx, span.Length).CopyTo(span);
                idx += span.Length;
            }
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(PagedKVCache));
}