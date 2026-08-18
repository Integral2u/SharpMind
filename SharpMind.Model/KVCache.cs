using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Stores cached Key and Value tensors for a single transformer layer.
/// Pre-allocated to <paramref name="maxSeqLen"/> to avoid O(log N) growth cost
/// and GC spikes during generation.
/// </summary>
public sealed class KVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim) : IKVCache
{
    private readonly Tensor<float> _keys = new(batchSize, numKvHeads, maxSeqLen, headDim);
    private readonly Tensor<float> _values = new(batchSize, numKvHeads, maxSeqLen, headDim);
    private readonly int _batchSize = batchSize;
    private readonly int _numKvHeads = numKvHeads;
    private readonly int _headDim = headDim;

    public Tensor<float> Keys => _keys;
    public Tensor<float> Values => _values;
    public int CurrentPosition { get; private set; } = 0;

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

    public int Length => CurrentPosition;
    public bool IsFull => CurrentPosition >= MaxSeqLen;

    public void Reset() => CurrentPosition = 0;

    public void Truncate(int length) => CurrentPosition = Math.Min(length, CurrentPosition);

    public void TrimToLast(int keep)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(keep);
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
            }
        }
        CurrentPosition = keep;
    }

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        int batch  = k.Shape[0];
        int seqLen = k.Shape[1];

        if (CurrentPosition + seqLen > MaxSeqLen)
            throw new InvalidOperationException(
                $"KVCache overflow: position {CurrentPosition} + seqLen {seqLen} exceeds capacity {MaxSeqLen}.");

        unsafe
        {
            uint rowBytes = (uint)headDim * sizeof(float);

            if (numKvHeads == 1)
            {
                long seqBlockBytes = (long)seqLen * headDim * sizeof(float);
                long dstCapBytes  = (long)(MaxSeqLen - CurrentPosition) * headDim * sizeof(float);
                for (int b = 0; b < batch; b++)
                {
                    float* srcK = k.DataPtr + (long)b * seqLen * headDim;
                    float* dstK = _keys.DataPtr + (long)b * MaxSeqLen * headDim + (long)CurrentPosition * headDim;
                    Buffer.MemoryCopy(srcK, dstK, dstCapBytes, seqBlockBytes);

                    float* srcV = v.DataPtr + (long)b * seqLen * headDim;
                    float* dstV = _values.DataPtr + (long)b * MaxSeqLen * headDim + (long)CurrentPosition * headDim;
                    Buffer.MemoryCopy(srcV, dstV, dstCapBytes, seqBlockBytes);
                }
            }
            else
            {
                for (int b = 0; b < batch; b++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        for (int h = 0; h < numKvHeads; h++)
                        {
                            float* srcK = k.DataPtr
                                + (long)b * (seqLen * numKvHeads * headDim)
                                + (long)s * (numKvHeads * headDim)
                                + (long)h * headDim;

                            float* dstK = _keys.DataPtr
                                + (long)b * (_numKvHeads * MaxSeqLen * headDim)
                                + (long)h * (MaxSeqLen * headDim)
                                + (long)(CurrentPosition + s) * headDim;

                            Unsafe.CopyBlock(dstK, srcK, rowBytes);

                            float* srcV = v.DataPtr
                                + (long)b * (seqLen * numKvHeads * headDim)
                                + (long)s * (numKvHeads * headDim)
                                + (long)h * headDim;

                            float* dstV = _values.DataPtr
                                + (long)b * (_numKvHeads * MaxSeqLen * headDim)
                                + (long)h * (MaxSeqLen * headDim)
                                + (long)(CurrentPosition + s) * headDim;

                            Unsafe.CopyBlock(dstV, srcV, rowBytes);
                        }
                    }
                }
            }
        }

        CurrentPosition += seqLen;
    }

    public object? Snapshot()
    {
        if (CurrentPosition == 0) return null;
        long activeLenLong = (long)_batchSize * _numKvHeads * CurrentPosition * _headDim;
        if (activeLenLong > int.MaxValue)
            throw new InvalidOperationException($"KVCache snapshot of {activeLenLong} elements overflows int.");
        int activeLen = (int)activeLenLong;
        var k = new float[activeLen];
        var v = new float[activeLen];
        _keys.Data[..activeLen].CopyTo(k);
        _values.Data[..activeLen].CopyTo(v);
        return (CurrentPosition, k, v);
    }

    public void Restore(object? snapshot)
    {
        if (snapshot is null) return;
        var (pos, k, v) = ((int, float[], float[]))snapshot;
        k.AsSpan().CopyTo(_keys.Data);
        v.AsSpan().CopyTo(_values.Data);
        CurrentPosition = pos;
    }

    public void Dispose()
    {
        _keys.Dispose();
        _values.Dispose();
    }
}
