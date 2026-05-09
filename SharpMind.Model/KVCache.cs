using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Stores cached Key and Value tensors for a single transformer layer.
/// This allows auto-regressive generation to avoid re-computing the entire sequence.
///
/// Memory strategy: starts at <paramref name="initialCapacity"/> tokens and doubles
/// up to <paramref name="maxSeqLen"/> as the conversation grows. This avoids the
/// O(MaxSeqLen) up-front cost for short conversations while still supporting the
/// full context length when needed.
/// </summary>
public sealed class KVCache : IDisposable
{
    private Tensor<float> _keys;
    private Tensor<float> _values;
    private readonly int _batchSize;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private int _allocatedCapacity;

    public Tensor<float> Keys => _keys;
    public Tensor<float> Values => _values;
    public int CurrentPosition { get; private set; }

    /// <summary>Hard upper bound — the model's true context limit.</summary>
    public int MaxSeqLen { get; }

    /// <summary>Currently allocated token capacity (may be less than MaxSeqLen).</summary>
    public int AllocatedCapacity => _allocatedCapacity;

    /// <param name="initialCapacity">
    /// Tokens to pre-allocate now. Defaults to 64 — enough for a first reply
    /// without wasting memory for the full context window.
    /// </param>
    public KVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim, int initialCapacity = 64)
    {
        _batchSize    = batchSize;
        _numKvHeads   = numKvHeads;
        _headDim      = headDim;
        MaxSeqLen     = maxSeqLen;

        // Clamp so we never over-allocate on tiny models.
        _allocatedCapacity = Math.Clamp(initialCapacity, 1, maxSeqLen);

        _keys   = new Tensor<float>(batchSize, numKvHeads, _allocatedCapacity, headDim);
        _values = new Tensor<float>(batchSize, numKvHeads, _allocatedCapacity, headDim);
        CurrentPosition = 0;
    }

    public int Length => CurrentPosition;
    public bool IsFull => CurrentPosition >= MaxSeqLen;

    public void Reset() => CurrentPosition = 0;

    /// <summary>
    /// Grows the internal tensors to <paramref name="newCapacity"/> tokens,
    /// preserving all cached K/V data in [0, CurrentPosition).
    /// No-op if already large enough.
    /// </summary>
    private void Grow(int newCapacity)
    {
        if (newCapacity <= _allocatedCapacity) return;
        newCapacity = Math.Min(newCapacity, MaxSeqLen);

        var newKeys   = new Tensor<float>(_batchSize, _numKvHeads, newCapacity, _headDim);
        var newValues = new Tensor<float>(_batchSize, _numKvHeads, newCapacity, _headDim);

        // Copy all cached positions into the new, larger tensors.
        unsafe
        {
            long rowBytes = (long)_headDim * sizeof(float);
            for (int b = 0; b < _batchSize; b++)
            {
                for (int h = 0; h < _numKvHeads; h++)
                {
                    for (int pos = 0; pos < CurrentPosition; pos++)
                    {
                        float* srcK = _keys.DataPtr
                            + (long)b * (_numKvHeads * _allocatedCapacity * _headDim)
                            + (long)h * (_allocatedCapacity * _headDim)
                            + (long)pos * _headDim;
                        float* dstK = newKeys.DataPtr
                            + (long)b * (_numKvHeads * newCapacity * _headDim)
                            + (long)h * (newCapacity * _headDim)
                            + (long)pos * _headDim;
                        Buffer.MemoryCopy(srcK, dstK, rowBytes, rowBytes);

                        float* srcV = _values.DataPtr
                            + (long)b * (_numKvHeads * _allocatedCapacity * _headDim)
                            + (long)h * (_allocatedCapacity * _headDim)
                            + (long)pos * _headDim;
                        float* dstV = newValues.DataPtr
                            + (long)b * (_numKvHeads * newCapacity * _headDim)
                            + (long)h * (newCapacity * _headDim)
                            + (long)pos * _headDim;
                        Buffer.MemoryCopy(srcV, dstV, rowBytes, rowBytes);
                    }
                }
            }
        }

        _keys.Dispose();
        _values.Dispose();
        _keys   = newKeys;
        _values = newValues;
        _allocatedCapacity = newCapacity;
    }

    public void TrimToLast(int keep)
    {
        if (keep < 0)
            throw new ArgumentOutOfRangeException(nameof(keep));
        if (keep >= CurrentPosition) return;

        int offset = CurrentPosition - keep;
        unsafe
        {
            long tokenStride = (long)_headDim * sizeof(float);

            for (int b = 0; b < _batchSize; b++)
            {
                for (int h = 0; h < _numKvHeads; h++)
                {
                    float* kPtr = _keys.DataPtr
                        + (long)b * (_numKvHeads * _allocatedCapacity * _headDim)
                        + (long)h * (_allocatedCapacity * _headDim);
                    float* vPtr = _values.DataPtr
                        + (long)b * (_numKvHeads * _allocatedCapacity * _headDim)
                        + (long)h * (_allocatedCapacity * _headDim);

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
            }
        }
        CurrentPosition = keep;
    }

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        int batch  = k.Shape[0];
        int seqLen = k.Shape[1];

        // Hard cap: refuse data beyond the model's true context limit.
        if (CurrentPosition + seqLen > MaxSeqLen)
            throw new InvalidOperationException(
                $"KVCache overflow: position {CurrentPosition} + seqLen {seqLen} exceeds capacity {MaxSeqLen}.");

        // Grow the backing tensors if this batch doesn't fit in the current allocation.
        if (CurrentPosition + seqLen > _allocatedCapacity)
        {
            int needed = CurrentPosition + seqLen;
            int doubled = _allocatedCapacity * 2;
            Grow(Math.Max(needed, doubled));
        }

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numKvHeads; h++)
                {
                    unsafe
                    {
                        float* srcPtr = k.DataPtr
                            + (long)b * (seqLen * numKvHeads * headDim)
                            + (long)s * (numKvHeads * headDim)
                            + (long)h * headDim;

                        float* dstPtr = _keys.DataPtr
                            + (long)b * (_numKvHeads * _allocatedCapacity * headDim)
                            + (long)h * (_allocatedCapacity * headDim)
                            + (long)(CurrentPosition + s) * headDim;

                        for (int d = 0; d < headDim; d++)
                            dstPtr[d] = srcPtr[d];

                        srcPtr = v.DataPtr
                            + (long)b * (seqLen * numKvHeads * headDim)
                            + (long)s * (numKvHeads * headDim)
                            + (long)h * headDim;

                        dstPtr = _values.DataPtr
                            + (long)b * (_numKvHeads * _allocatedCapacity * headDim)
                            + (long)h * (_allocatedCapacity * headDim)
                            + (long)(CurrentPosition + s) * headDim;

                        for (int d = 0; d < headDim; d++)
                            dstPtr[d] = srcPtr[d];
                    }
                }
            }
        }

        CurrentPosition += seqLen;
    }

    public void Dispose()
    {
        _keys.Dispose();
        _values.Dispose();
    }
}