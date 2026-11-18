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
/// </summary>
public sealed class MlaKvCache(int batchSize, int maxSeqLen, int latentDim) : IKVCache
{
    private readonly Tensor<float> _latent = new(batchSize, maxSeqLen, latentDim);
    private readonly int _batch = batchSize;

    /// <summary>Floats cached per position: <c>kv_lora_rank + rope</c>.</summary>
    public int LatentDim { get; } = latentDim;

    public int CurrentPosition { get; private set; }
    public int MaxSeqLen { get; } = maxSeqLen;
    public int Length => CurrentPosition;
    public bool IsFull => CurrentPosition >= MaxSeqLen;
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

        if (CurrentPosition + seqLen > MaxSeqLen)
            throw new InvalidOperationException(
                $"MlaKvCache overflow: position {CurrentPosition} + seqLen {seqLen} exceeds capacity {MaxSeqLen}.");
        if (headDim != LatentDim || numKvHeads != 1)
            throw new InvalidOperationException(
                $"MlaKvCache takes one latent head of {LatentDim} floats; got {numKvHeads} head(s) of {headDim}.");

        unsafe
        {
            long rowBytes = (long)seqLen * LatentDim * sizeof(float);
            long dstCapBytes = (long)(MaxSeqLen - CurrentPosition) * LatentDim * sizeof(float);
            for (int b = 0; b < batch; b++)
            {
                float* src = k.DataPtr + (long)b * seqLen * LatentDim;
                float* dst = _latent.DataPtr + ((long)b * MaxSeqLen + CurrentPosition) * LatentDim;
                Buffer.MemoryCopy(src, dst, dstCapBytes, rowBytes);
            }
        }

        CurrentPosition += seqLen;
    }

    public void Reset() => CurrentPosition = 0;

    public void Truncate(int length) => CurrentPosition = Math.Min(length, CurrentPosition);

    public void TrimToLast(int keep)
    {
        if (keep >= CurrentPosition) return;
        if (keep <= 0) { CurrentPosition = 0; return; }

        unsafe
        {
            int start = CurrentPosition - keep;
            for (int b = 0; b < _latent.Shape[0]; b++)
            {
                float* rowBase = _latent.DataPtr + (long)b * MaxSeqLen * LatentDim;
                Buffer.MemoryCopy(rowBase + (long)start * LatentDim, rowBase,
                    (long)MaxSeqLen * LatentDim * sizeof(float),
                    (long)keep * LatentDim * sizeof(float));
            }
        }
        CurrentPosition = keep;
    }

    /// <summary>Snapshot as <c>(position, latent rows)</c>, the tag-2 form the session writer understands.</summary>
    public object? Snapshot()
    {
        if (CurrentPosition == 0) return null;
        int batch = _latent.Shape[0];
        var data = new float[(long)batch * CurrentPosition * LatentDim];
        unsafe
        {
            fixed (float* dst = data)
            {
                for (int b = 0; b < batch; b++)
                {
                    Buffer.MemoryCopy(
                        _latent.DataPtr + (long)b * MaxSeqLen * LatentDim,
                        dst + (long)b * CurrentPosition * LatentDim,
                        (long)CurrentPosition * LatentDim * sizeof(float),
                        (long)CurrentPosition * LatentDim * sizeof(float));
                }
            }
        }
        return (CurrentPosition, data);
    }

    public void Restore(object? snapshot)
    {
        if (snapshot is null) { CurrentPosition = 0; return; }
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
        CurrentPosition = position;
    }

    public void Dispose() => _latent.Dispose();
}
