using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Stores cached Key and Value tensors for a single transformer layer.
/// Pre-allocated to <paramref name="maxSeqLen"/> to avoid O(log N) growth cost
/// and GC spikes during generation.
///
/// Each of the <paramref name="batchSize"/> rows tracks its own length. A
/// single-sequence caller never notices (one row, one length), but a batched
/// decode needs it: sequences at different positions append at different slots
/// and attend over different amounts of history in the same forward pass.
/// </summary>
public sealed class KVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim) : IKVCache
{
    private readonly Tensor<float> _keys = new(batchSize, numKvHeads, maxSeqLen, headDim);
    private readonly Tensor<float> _values = new(batchSize, numKvHeads, maxSeqLen, headDim);
    private readonly int _batchSize = batchSize;
    private readonly int _numKvHeads = numKvHeads;
    private readonly int _headDim = headDim;
    private readonly int[] _rowLengths = new int[batchSize];
    private int _maxLength;

    public Tensor<float> Keys => _keys;
    public Tensor<float> Values => _values;

    /// <summary>Longest row; equals the cache length for every single-sequence use.</summary>
    public int CurrentPosition => _maxLength;

    public int MaxSeqLen { get; } = maxSeqLen;

    public bool IsContiguous => true;

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead)
        => _keys.DataPtr
            + (long)batchIdx * (_numKvHeads * MaxSeqLen * _headDim)
            + (long)kvHead * (MaxSeqLen * _headDim)
            + (long)position * _headDim;

    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead)
        => _values.DataPtr
            + (long)batchIdx * (_numKvHeads * MaxSeqLen * _headDim)
            + (long)kvHead * (MaxSeqLen * _headDim)
            + (long)position * _headDim;

    public int Length => _maxLength;
    public int RowCount => _batchSize;
    public bool SupportsRaggedRows => true;
    public int LengthForRow(int row) => _rowLengths[row];
    public bool IsFull => _maxLength >= MaxSeqLen;

    public void Reset()
    {
        Array.Clear(_rowLengths);
        _maxLength = 0;
    }

    public void Truncate(int length)
    {
        for (int b = 0; b < _batchSize; b++)
            if (_rowLengths[b] > length) _rowLengths[b] = length;
        RecomputeMax();
    }

    /// <summary>Drops trailing entries from one row only.</summary>
    public void TruncateRow(int row, int length)
    {
        if ((uint)row >= (uint)_batchSize)
            throw new ArgumentOutOfRangeException(nameof(row), row, $"Row must be in [0, {_batchSize}).");
        if (length < 0) length = 0;
        if (_rowLengths[row] > length) _rowLengths[row] = length;
        RecomputeMax();
    }

    public void TrimToLast(int keep)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(keep);

        unsafe
        {
            long tokenStride = (long)_headDim * sizeof(float);

            for (int b = 0; b < _batchSize; b++)
            {
                int length = _rowLengths[b];
                if (keep >= length) continue;

                int offset = length - keep;
                for (int h = 0; h < _numKvHeads; h++)
                {
                    float* kPtr = _keys.DataPtr
                        + (long)b * (_numKvHeads * MaxSeqLen * _headDim)
                        + (long)h * (MaxSeqLen * _headDim);
                    float* vPtr = _values.DataPtr
                        + (long)b * (_numKvHeads * MaxSeqLen * _headDim)
                        + (long)h * (MaxSeqLen * _headDim);

                    // Move the retained window [offset, offset+keep) to [0, keep).
                    for (int i = 0; i < keep; i++)
                    {
                        float* srcK = kPtr + (long)(offset + i) * _headDim;
                        float* dstK = kPtr + (long)i * _headDim;
                        Buffer.MemoryCopy(srcK, dstK, tokenStride, tokenStride);

                        float* srcV = vPtr + (long)(offset + i) * _headDim;
                        float* dstV = vPtr + (long)i * _headDim;
                        Buffer.MemoryCopy(srcV, dstV, tokenStride, tokenStride);
                    }
                }

                _rowLengths[b] = keep;
            }
        }
        RecomputeMax();
    }

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        int batch = k.Shape[0];
        int seqLen = k.Shape[1];
        if (batch > _batchSize)
            throw new InvalidOperationException(
                $"KVCache holds {_batchSize} row(s) but the update carries {batch}.");

        for (int b = 0; b < batch; b++)
        {
            long rowStart = (long)b * seqLen * numKvHeads * headDim;
            int rowLength = seqLen * numKvHeads * headDim;
            UpdateRow(b,
                k.Data.Slice((int)rowStart, rowLength),
                v.Data.Slice((int)rowStart, rowLength),
                numKvHeads, headDim);
        }
    }

    /// <summary>
    /// Appends one row at that row's own length. The span is the row's
    /// <c>[seqLen, numKvHeads, headDim]</c> block.
    /// </summary>
    public unsafe void UpdateRow(int row, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int numKvHeads, int headDim)
    {
        if ((uint)row >= (uint)_batchSize)
            throw new ArgumentOutOfRangeException(nameof(row), row, $"Row must be in [0, {_batchSize}).");
        if (headDim != _headDim)
            throw new InvalidOperationException(
                $"KVCache stores {_headDim}-wide heads but the update carries {headDim}.");

        int position = _rowLengths[row];
        int seqLen = k.Length / Math.Max(1, numKvHeads * headDim);
        if (position + seqLen > MaxSeqLen)
            throw new InvalidOperationException(
                $"KVCache overflow: row {row} at position {position} + seqLen {seqLen} exceeds capacity {MaxSeqLen}.");

        fixed (float* kPtr = k)
        fixed (float* vPtr = v)
        {
            if (numKvHeads == 1)
            {
                long seqBlockBytes = (long)seqLen * headDim * sizeof(float);
                long dstCapBytes  = (long)(MaxSeqLen - position) * headDim * sizeof(float);
                float* dstK = _keys.DataPtr + (long)row * MaxSeqLen * headDim + (long)position * headDim;
                float* dstV = _values.DataPtr + (long)row * MaxSeqLen * headDim + (long)position * headDim;
                Buffer.MemoryCopy(kPtr, dstK, dstCapBytes, seqBlockBytes);
                Buffer.MemoryCopy(vPtr, dstV, dstCapBytes, seqBlockBytes);
            }
            else
            {
                uint rowBytes = (uint)headDim * sizeof(float);
                for (int s = 0; s < seqLen; s++)
                {
                    for (int h = 0; h < numKvHeads; h++)
                    {
                        float* srcK = kPtr + (long)s * (numKvHeads * headDim) + (long)h * headDim;
                        float* dstK = _keys.DataPtr
                            + (long)row * (_numKvHeads * MaxSeqLen * headDim)
                            + (long)h * (MaxSeqLen * headDim)
                            + (long)(position + s) * headDim;
                        Unsafe.CopyBlock(dstK, srcK, rowBytes);

                        float* srcV = vPtr + (long)s * (numKvHeads * headDim) + (long)h * headDim;
                        float* dstV = _values.DataPtr
                            + (long)row * (_numKvHeads * MaxSeqLen * headDim)
                            + (long)h * (MaxSeqLen * headDim)
                            + (long)(position + s) * headDim;
                        Unsafe.CopyBlock(dstV, srcV, rowBytes);
                    }
                }
            }
        }

        _rowLengths[row] = position + seqLen;
        if (_rowLengths[row] > _maxLength) _maxLength = _rowLengths[row];
    }

    /// <summary>Copies one row's keys and values onto another row and clears the source.</summary>
    public unsafe void MoveRow(int from, int to)
    {
        if ((uint)from >= (uint)_batchSize || (uint)to >= (uint)_batchSize)
            throw new ArgumentOutOfRangeException(nameof(from), "Row index outside the cache.");
        if (from == to) return;

        long rowFloats = (long)_numKvHeads * MaxSeqLen * _headDim;
        long rowBytes = rowFloats * sizeof(float);
        Buffer.MemoryCopy(_keys.DataPtr + (long)from * rowFloats,
                          _keys.DataPtr + (long)to * rowFloats, rowBytes, rowBytes);
        Buffer.MemoryCopy(_values.DataPtr + (long)from * rowFloats,
                          _values.DataPtr + (long)to * rowFloats, rowBytes, rowBytes);
        _rowLengths[to] = _rowLengths[from];
        _rowLengths[from] = 0;
        RecomputeMax();
    }

    private void RecomputeMax()
    {
        int max = 0;
        foreach (int length in _rowLengths)
            if (length > max) max = length;
        _maxLength = max;
    }

    /// <summary>Empties one row without touching the others.</summary>
    public void ClearRow(int row)
    {
        if ((uint)row >= (uint)_batchSize)
            throw new ArgumentOutOfRangeException(nameof(row), row, $"Row must be in [0, {_batchSize}).");
        _rowLengths[row] = 0;
        RecomputeMax();
    }

    public object? Snapshot()
    {
        int position = RequireUniformLengths();
        if (position == 0) return null;
        long activeLenLong = (long)_batchSize * _numKvHeads * position * _headDim;
        if (activeLenLong > int.MaxValue)
            throw new InvalidOperationException($"KVCache snapshot of {activeLenLong} elements overflows int.");
        int activeLen = (int)activeLenLong;
        var k = new float[activeLen];
        var v = new float[activeLen];
        int destination = 0;
        int rowsPerHead = position * _headDim;
        for (int b = 0; b < _batchSize; b++)
        {
            for (int h = 0; h < _numKvHeads; h++)
            {
                int source = (b * _numKvHeads + h) * MaxSeqLen * _headDim;
                _keys.Data.Slice(source, rowsPerHead).CopyTo(k.AsSpan(destination, rowsPerHead));
                _values.Data.Slice(source, rowsPerHead).CopyTo(v.AsSpan(destination, rowsPerHead));
                destination += rowsPerHead;
            }
        }
        return (position, k, v);
    }

    public void Restore(object? snapshot)
    {
        if (snapshot is null)
        {
            Reset();
            return;
        }
        var (pos, k, v) = ((int, float[], float[]))snapshot;
        if ((uint)pos > (uint)MaxSeqLen)
            throw new InvalidDataException($"KVCache snapshot position {pos} exceeds capacity {MaxSeqLen}.");
        int rowsPerHead = checked(pos * _headDim);
        int expectedLength = checked(_batchSize * _numKvHeads * rowsPerHead);
        if (k.Length != expectedLength || v.Length != expectedLength)
            throw new InvalidDataException("KVCache snapshot dimensions do not match this cache.");

        int source = 0;
        for (int b = 0; b < _batchSize; b++)
        {
            for (int h = 0; h < _numKvHeads; h++)
            {
                int destination = (b * _numKvHeads + h) * MaxSeqLen * _headDim;
                k.AsSpan(source, rowsPerHead).CopyTo(_keys.Data.Slice(destination, rowsPerHead));
                v.AsSpan(source, rowsPerHead).CopyTo(_values.Data.Slice(destination, rowsPerHead));
                source += rowsPerHead;
            }
        }
        Array.Fill(_rowLengths, pos);
        _maxLength = pos;
    }

    /// <summary>
    /// A snapshot has one position for every row, so rows of different lengths
    /// cannot be represented in the tag-1 format. Session save is single-row;
    /// refuse rather than silently flatten a ragged batch.
    /// </summary>
    private int RequireUniformLengths()
    {
        int position = _rowLengths.Length > 0 ? _rowLengths[0] : 0;
        for (int b = 1; b < _batchSize; b++)
            if (_rowLengths[b] != position)
                throw new InvalidOperationException(
                    "Cannot snapshot a KVCache whose rows hold different lengths; " +
                    "the snapshot format stores one position per cache.");
        return position;
    }

    public void Dispose()
    {
        _keys.Dispose();
        _values.Dispose();
    }
}
