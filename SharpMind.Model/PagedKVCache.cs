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

        long stridePageLong = (long)numKvHeads * pageSize * headDim;
        if (stridePageLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(pageSize), $"PagedKV page stride {stridePageLong} overflows int.");
        long strideHeadLong = (long)pageSize * headDim;
        if (strideHeadLong > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(headDim), $"PagedKV head stride {strideHeadLong} overflows int.");
        _stridePage = (int)stridePageLong;
        _strideHead = (int)strideHeadLong;

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

    public unsafe void TrimToLast(int keepTokens)
    {
        ThrowIfDisposed();
        if (keepTokens >= _currentPosition) return;
        int offset = _currentPosition - keepTokens;
        uint rowBytes = (uint)_headDim * sizeof(float);

        for (int b = 0; b < _batchSize; b++)
        {
            long batchBase = (long)b * _numPages * _stridePage;
            for (int s = 0; s < keepTokens; s++)
            {
                var (srcPage, srcOff) = GetPageInfo(offset + s);
                var (dstPage, dstOff) = GetPageInfo(s);

                float* srcBaseK = _keys.DataPtr + batchBase + (long)srcPage * _stridePage;
                float* dstBaseK = _keys.DataPtr + batchBase + (long)dstPage * _stridePage;
                float* srcBaseV = _values.DataPtr + batchBase + (long)srcPage * _stridePage;
                float* dstBaseV = _values.DataPtr + batchBase + (long)dstPage * _stridePage;

                for (int h = 0; h < _numKvHeads; h++)
                {
                    Unsafe.CopyBlock(
                        dstBaseK + (long)h * _strideHead + (long)dstOff * _headDim,
                        srcBaseK + (long)h * _strideHead + (long)srcOff * _headDim,
                        rowBytes);
                    Unsafe.CopyBlock(
                        dstBaseV + (long)h * _strideHead + (long)dstOff * _headDim,
                        srcBaseV + (long)h * _strideHead + (long)srcOff * _headDim,
                        rowBytes);
                }
            }
        }
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
        int activePages = (_currentPosition + _pageSize - 1) / _pageSize;
        long totalFloatsLong = (long)activePages * _stridePage * _batchSize;
        if (totalFloatsLong * 2 > int.MaxValue)
            throw new InvalidOperationException(
                $"PagedKVCache snapshot of {totalFloatsLong * 2} floats overflows int (position {_currentPosition}/{_maxSeqLen}).");
        int batchFloats = checked(activePages * _stridePage);
        int totalFloats = (int)totalFloatsLong;
        var data = new float[totalFloats * 2];

        for (int b = 0; b < _batchSize; b++)
        {
            int srcOff = b * _numPages * _stridePage;
            int dstOff = b * batchFloats;
            _keys.Data.Slice(srcOff, batchFloats).CopyTo(data.AsSpan(dstOff, batchFloats));
            _values.Data.Slice(srcOff, batchFloats).CopyTo(data.AsSpan(totalFloats + dstOff, batchFloats));
        }
        return (_currentPosition, data);
    }

    public void Restore(object? snapshot)
    {
        ThrowIfDisposed();
        if (snapshot is null) return;
        var (pos, data) = ((int, float[]))snapshot;
        int activePages = (pos + _pageSize - 1) / _pageSize;
        int batchFloats = activePages * _stridePage;
        int totalFloats = batchFloats * _batchSize;

        for (int b = 0; b < _batchSize; b++)
        {
            int dstOff = b * _numPages * _stridePage;
            int srcOff = b * batchFloats;
            data.AsSpan(srcOff, batchFloats).CopyTo(_keys.Data.Slice(dstOff, batchFloats));
            data.AsSpan(totalFloats + srcOff, batchFloats).CopyTo(_values.Data.Slice(dstOff, batchFloats));
        }
        _currentPosition = pos;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(PagedKVCache));
}
