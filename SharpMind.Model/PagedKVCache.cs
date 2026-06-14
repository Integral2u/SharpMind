using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Paged KV Cache — uses fixed-size pages instead of a single contiguous buffer.
///
/// Memory layout (single flat allocation per K/V):
///   [batch, numPages, numKvHeads, pageSize, headDim]
///
/// This eliminates the per-head-page Tensor allocations that the previous
/// <c>Tensor&lt;float&gt;[,]</c> design required, and enables efficient
/// <see cref="Unsafe.CopyBlock"/>-based updates instead of element-wise loops.
/// </summary>
public sealed class PagedKVCache : IDisposable
{
    public const int DefaultPageSize = 32;

    private readonly int _batchSize;
    private readonly int _numKvHeads;
    private readonly int _maxSeqLen;
    private readonly int _headDim;
    private readonly int _pageSize;
    private readonly int _numPages;

    // Single flat allocations — one per K and V.
    // Layout: [batchSize, numPages, numKvHeads, pageSize, headDim]
    private readonly Tensor<float> _keys;
    private readonly Tensor<float> _values;

    // Stride helpers (in floats) for fast pointer arithmetic.
    private readonly int _stridePage;  // numKvHeads * pageSize * headDim
    private readonly int _strideHead;  // pageSize * headDim

    private int _currentPosition;
    private bool _disposed;

    public PagedKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim, int pageSize = DefaultPageSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(numKvHeads);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxSeqLen);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(headDim);

        _batchSize  = batchSize;
        _numKvHeads = numKvHeads;
        _maxSeqLen  = maxSeqLen;
        _headDim    = headDim;
        _pageSize   = pageSize;
        _numPages   = (maxSeqLen + pageSize - 1) / pageSize;

        _stridePage = numKvHeads * pageSize * headDim;
        _strideHead = pageSize * headDim;

        _keys   = new Tensor<float>(batchSize, _numPages, numKvHeads, pageSize, headDim);
        _values = new Tensor<float>(batchSize, _numPages, numKvHeads, pageSize, headDim);
    }

    public int Length => _currentPosition;
    public int PageSize => _pageSize;
    public int MaxSeqLen => _maxSeqLen;
    public bool IsFull => _currentPosition >= _maxSeqLen;

    public (int pageIdx, int offset) GetPageInfo(int position)
    {
        int pageIdx = position / _pageSize;
        int offset  = position % _pageSize;
        return (pageIdx, offset);
    }

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead)
    {
        var (pageIdx, offset) = GetPageInfo(position);
        return _keys.DataPtr
            + (long)batchIdx * _numPages * _stridePage
            + (long)pageIdx * _stridePage
            + (long)kvHead * _strideHead
            + (long)offset * _headDim;
    }

    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead)
    {
        var (pageIdx, offset) = GetPageInfo(position);
        return _values.DataPtr
            + (long)batchIdx * _numPages * _stridePage
            + (long)pageIdx * _stridePage
            + (long)kvHead * _strideHead
            + (long)offset * _headDim;
    }

    public unsafe void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        ThrowIfDisposed();

        int batch  = k.Shape[0];
        int seqLen = k.Shape[1];

        if (_currentPosition + seqLen > _maxSeqLen)
            throw new InvalidOperationException(
                $"KV cache overflow: {_currentPosition} + {seqLen} > {_maxSeqLen}");

        uint rowBytes = (uint)headDim * sizeof(float);

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int dstPos = _currentPosition + s;
                var (pageIdx, offset) = GetPageInfo(dstPos);

                float* pageBaseK = _keys.DataPtr
                    + (long)b * _numPages * _stridePage
                    + (long)pageIdx * _stridePage;
                float* pageBaseV = _values.DataPtr
                    + (long)b * _numPages * _stridePage
                    + (long)pageIdx * _stridePage;

                float* srcRowK = k.DataPtr
                    + (long)b * seqLen * numKvHeads * headDim
                    + (long)s * numKvHeads * headDim;
                float* srcRowV = v.DataPtr
                    + (long)b * seqLen * numKvHeads * headDim
                    + (long)s * numKvHeads * headDim;

                for (int h = 0; h < numKvHeads; h++)
                {
                    float* dstK = pageBaseK + (long)h * _strideHead + (long)offset * headDim;
                    float* dstV = pageBaseV + (long)h * _strideHead + (long)offset * headDim;
                    Unsafe.CopyBlock(dstK, srcRowK + (long)h * headDim, rowBytes);
                    Unsafe.CopyBlock(dstV, srcRowV + (long)h * headDim, rowBytes);
                }
            }
        }

        _currentPosition += seqLen;
    }

    public void Reset()
    {
        ThrowIfDisposed();
        _currentPosition = 0;
    }

    public void Truncate(int length)
    {
        ThrowIfDisposed();
        _currentPosition = Math.Min(length, _currentPosition);
    }

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
        _keys.Dispose();
        _values.Dispose();
    }

    public object? Snapshot()
    {
        ThrowIfDisposed();
        if (_currentPosition == 0) return null;
        var data = new float[_keys.ElementCount + _values.ElementCount];
        _keys.Data.CopyTo(data.AsSpan(0, _keys.ElementCount));
        _values.Data.CopyTo(data.AsSpan(_keys.ElementCount, _values.ElementCount));
        return (_currentPosition, data);
    }

    public void Restore(object? snapshot)
    {
        ThrowIfDisposed();
        if (snapshot is null) return;
        var (pos, data) = ((int, float[]))snapshot;
        _currentPosition = pos;
        data.AsSpan(0, _keys.ElementCount).CopyTo(_keys.Data);
        data.AsSpan(_keys.ElementCount, _values.ElementCount).CopyTo(_values.Data);
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(PagedKVCache));
}
