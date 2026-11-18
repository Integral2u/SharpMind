using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// KV cache for Multi-head Latent Attention. MLA does not cache a key and a
/// value per head: it caches one compressed latent per position,
/// <c>kv_lora_rank + qk_rope_head_dim</c> wide (576 in DeepSeek V2/V3), and
/// rebuilds K/V from it — or, on the decode path, absorbs the projection into
/// the query and never rebuilds them at all.
///
/// That is the whole point of the architecture: at 32K tokens this cache is
/// 576 floats per position per layer against 2 × 16 heads × 192/128 for the
/// expanded form (~2.3 KB vs ~20.5 KB per position per layer).
///
/// The rows are laid out as a single "head" of <c>latentDim</c>, which is what
/// <see cref="IKVCache.Update"/>'s one-head branch produces and what
/// <see cref="GetKeyPtr"/> returns; there is no separate value tensor, so
/// <see cref="GetValuePtr"/> hands back the same row. Callers that need values
/// (nothing does for MLA) must project them from the latent.
///
/// Like <see cref="KVCache"/>, every row tracks its own length so a batched
/// decode can hold sequences at different positions.
/// </summary>
public sealed class MlaKvCache(int batchSize, int maxSeqLen, int latentDim) : IKVCache
{
    private readonly Tensor<float> _latent = new(batchSize, maxSeqLen, latentDim);
    private readonly int _batch = batchSize;
    private readonly int[] _rowLengths = new int[batchSize];
    private int _maxLength;

    /// <summary>Floats cached per position: <c>kv_lora_rank + rope</c>.</summary>
    public int LatentDim { get; } = latentDim;

    /// <summary>Longest row; equals the cache length for every single-sequence use.</summary>
    public int CurrentPosition => _maxLength;
    public int MaxSeqLen { get; } = maxSeqLen;
    public int Length => _maxLength;
    public int RowCount => _batch;
    public bool SupportsRaggedRows => true;
    public int LengthForRow(int row) => _rowLengths[row];
    public bool IsFull => _maxLength >= MaxSeqLen;
    public bool IsContiguous => true;

    /// <summary>Raw latent rows, [batch, maxSeqLen, latentDim].</summary>
    public Tensor<float> Latent => _latent;

    /// <summary>
    /// Returns a pointer to position <paramref name="position"/> of
    /// <paramref name="batchIdx"/>. Both are validated: the caller supplies them
    /// (they come from loop counters and window math in the attention paths), and
    /// an out-of-range pair would otherwise read past the latent buffer instead of
    /// failing. Only one latent head exists, so <paramref name="kvHead"/> must be 0.
    /// </summary>
    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead)
    {
        ValidateSlot(batchIdx, position, kvHead);
        return _latent.DataPtr + ((long)batchIdx * MaxSeqLen + position) * LatentDim;
    }

    /// <inheritdoc cref="GetKeyPtr"/> — MLA keeps no separate values.
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead) =>
        GetKeyPtr(batchIdx, position, kvHead);

    private void ValidateSlot(int batchIdx, int position, int kvHead)
    {
        if ((uint)batchIdx >= (uint)_batch)
            throw new ArgumentOutOfRangeException(nameof(batchIdx), batchIdx, $"Batch index must be in [0, {_batch}).");
        if ((uint)position >= (uint)MaxSeqLen)
            throw new ArgumentOutOfRangeException(nameof(position), position, $"Position must be in [0, {MaxSeqLen}).");
        if (kvHead != 0)
            throw new ArgumentOutOfRangeException(nameof(kvHead), kvHead, "MLA caches a single latent head.");
    }

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        int batch = k.Shape[0];
        int seqLen = k.Shape[1];
        if (batch > _batch)
            throw new InvalidOperationException(
                $"MlaKvCache holds {_batch} row(s) but the update carries {batch}.");

        for (int b = 0; b < batch; b++)
            UpdateRow(b, k.Data.Slice(b * seqLen * LatentDim, seqLen * LatentDim),
                         v.Data.Slice(b * seqLen * LatentDim, seqLen * LatentDim),
                         numKvHeads, headDim);
    }

    /// <summary>Appends one row's <c>[seqLen, latentDim]</c> latent block at that row's own length.</summary>
    public unsafe void UpdateRow(int row, ReadOnlySpan<float> k, ReadOnlySpan<float> v, int numKvHeads, int headDim)
    {
        if ((uint)row >= (uint)_batch)
            throw new ArgumentOutOfRangeException(nameof(row), row, $"Row must be in [0, {_batch}).");
        if (headDim != LatentDim || numKvHeads != 1)
            throw new InvalidOperationException(
                $"MlaKvCache takes one latent head of {LatentDim} floats; got {numKvHeads} head(s) of {headDim}.");

        int position = _rowLengths[row];
        int seqLen = k.Length / LatentDim;
        if (position + seqLen > MaxSeqLen)
            throw new InvalidOperationException(
                $"MlaKvCache overflow: row {row} at position {position} + seqLen {seqLen} exceeds capacity {MaxSeqLen}.");

        fixed (float* src = k)
        {
            long rowBytes = (long)seqLen * LatentDim * sizeof(float);
            long dstCapBytes = (long)(MaxSeqLen - position) * LatentDim * sizeof(float);
            float* dst = _latent.DataPtr + ((long)row * MaxSeqLen + position) * LatentDim;
            Buffer.MemoryCopy(src, dst, dstCapBytes, rowBytes);
        }

        _rowLengths[row] = position + seqLen;
        if (_rowLengths[row] > _maxLength) _maxLength = _rowLengths[row];
    }

    /// <summary>Copies one row's latent onto another row and clears the source.</summary>
    public unsafe void MoveRow(int from, int to)
    {
        if ((uint)from >= (uint)_batch || (uint)to >= (uint)_batch)
            throw new ArgumentOutOfRangeException(nameof(from), "Row index outside the cache.");
        if (from == to) return;

        long rowFloats = (long)MaxSeqLen * LatentDim;
        long rowBytes = rowFloats * sizeof(float);
        Buffer.MemoryCopy(_latent.DataPtr + (long)from * rowFloats,
                          _latent.DataPtr + (long)to * rowFloats, rowBytes, rowBytes);
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

    public void Reset()
    {
        Array.Clear(_rowLengths);
        _maxLength = 0;
    }

    public void Truncate(int length)
    {
        for (int b = 0; b < _batch; b++)
            if (_rowLengths[b] > length) _rowLengths[b] = length;
        RecomputeMax();
    }

    public void TrimToLast(int keep)
    {
        unsafe
        {
            for (int b = 0; b < _batch; b++)
            {
                int length = _rowLengths[b];
                if (keep >= length) continue;
                if (keep <= 0) { _rowLengths[b] = 0; continue; }

                int start = length - keep;
                float* rowBase = _latent.DataPtr + (long)b * MaxSeqLen * LatentDim;
                Buffer.MemoryCopy(rowBase + (long)start * LatentDim, rowBase,
                    (long)MaxSeqLen * LatentDim * sizeof(float),
                    (long)keep * LatentDim * sizeof(float));
                _rowLengths[b] = keep;
            }
        }
        RecomputeMax();
    }

    /// <summary>Snapshot as <c>(position, latent rows)</c>, the tag-2 form the session writer understands.</summary>
    public object? Snapshot()
    {
        int position = RequireUniformLengths();
        if (position == 0) return null;
        int batch = _latent.Shape[0];
        var data = new float[(long)batch * position * LatentDim];
        unsafe
        {
            fixed (float* dst = data)
            {
                for (int b = 0; b < batch; b++)
                {
                    Buffer.MemoryCopy(
                        _latent.DataPtr + (long)b * MaxSeqLen * LatentDim,
                        dst + (long)b * position * LatentDim,
                        (long)position * LatentDim * sizeof(float),
                        (long)position * LatentDim * sizeof(float));
                }
            }
        }
        return (position, data);
    }

    public void Restore(object? snapshot)
    {
        if (snapshot is null) { Reset(); return; }
        var (position, data) = ((int, float[]))snapshot;
        int batch = _latent.Shape[0];
        long rows = (long)batch * position * LatentDim;
        if (data.Length < rows)
            throw new InvalidDataException($"MLA cache snapshot holds {data.Length} floats, {rows} needed.");

        unsafe
        {
            fixed (float* src = data)
            {
                for (int b = 0; b < batch; b++)
                {
                    Buffer.MemoryCopy(
                        src + (long)b * position * LatentDim,
                        _latent.DataPtr + (long)b * MaxSeqLen * LatentDim,
                        (long)position * LatentDim * sizeof(float),
                        (long)position * LatentDim * sizeof(float));
                }
            }
        }
        Array.Fill(_rowLengths, position);
        _maxLength = position;
    }

    /// <summary>See <see cref="KVCache"/>: the snapshot format stores one position for every row.</summary>
    private int RequireUniformLengths()
    {
        int position = _rowLengths.Length > 0 ? _rowLengths[0] : 0;
        for (int b = 1; b < _batch; b++)
            if (_rowLengths[b] != position)
                throw new InvalidOperationException(
                    "Cannot snapshot an MlaKvCache whose rows hold different lengths; " +
                    "the snapshot format stores one position per cache.");
        return position;
    }

    public void Dispose() => _latent.Dispose();
}
