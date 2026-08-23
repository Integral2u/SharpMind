using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpMind.Core.Memory;

/// <summary>
/// A paged array that can hold more than <see cref="int.MaxValue"/> elements.
/// Pages are int-sized <see cref="ArrayPool{T}.Shared"/> segments.
/// Single-element access via long index; block iteration via int-sized spans.
/// </summary>
public sealed class BigArray<T> : IDisposable where T : unmanaged
{
    private const int PageSize = 1024 * 1024; // 1M elements
    private readonly int _pageCount;
    private readonly long _length;
    private readonly T[][] _pages;

    public long Length => _length;
    public int PageSizeBits => PageSize;

    public BigArray(long length)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        _length = length;
        _pageCount = (int)((length + PageSize - 1) / PageSize);
        _pages = new T[_pageCount][];
        for (int i = 0; i < _pageCount; i++)
        {
            int pageSize = GetPageSize(i);
            _pages[i] = ArrayPool<T>.Shared.Rent(pageSize);
            Array.Clear(_pages[i], 0, pageSize);
        }
    }

    public T this[long index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            if ((ulong)index >= (ulong)_length) throw new IndexOutOfRangeException();
            int pageIndex = (int)(index / PageSize);
            int offset = (int)(index % PageSize);
            return _pages[pageIndex][offset];
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            if ((ulong)index >= (ulong)_length) throw new IndexOutOfRangeException();
            int pageIndex = (int)(index / PageSize);
            int offset = (int)(index % PageSize);
            _pages[pageIndex][offset] = value;
        }
    }

    /// <summary>
    /// Returns a span over a single page for the given block index.
    /// Block index = flat index / PageSize. The span is int-sized.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> GetBlock(int blockIndex, out int blockOffset, out int blockCount)
    {
        if ((uint)blockIndex >= (uint)_pageCount) throw new ArgumentOutOfRangeException(nameof(blockIndex));
        blockCount = GetPageSize(blockIndex);
        blockOffset = 0;
        return _pages[blockIndex];
    }

    /// <summary>
    /// Returns the total number of int-sized blocks.
    /// </summary>
    public int BlockCount => _pageCount;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int GetPageSize(int pageIndex)
    {
        if (pageIndex < _pageCount - 1) return PageSize;
        long remaining = _length - (long)pageIndex * PageSize;
        return (int)Math.Min(remaining, PageSize);
    }

    public void Dispose()
    {
        for (int i = 0; i < _pageCount; i++)
        {
            if (_pages[i] is not null)
            {
                ArrayPool<T>.Shared.Return(_pages[i]);
                _pages[i] = null!;
            }
        }
    }
}
