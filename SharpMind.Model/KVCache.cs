using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Stores cached Key and Value tensors for a single transformer layer.
/// Pre-allocated to <paramref name="maxSeqLen"/> to avoid O(log N) growth cost
/// and GC spikes during generation.
/// </summary>
public sealed class KVCache : IKVCache
{
    private Tensor<float> _keys;
    private Tensor<float> _values;
    private readonly int _batchSize;
    private readonly int _numKvHeads;
    private readonly int _headDim;

    public Tensor<float> Keys => _keys;
    public Tensor<float> Values => _values;
    public int CurrentPosition { get; private set; }

    public int MaxSeqLen { get; }

    public int AllocatedCapacity => MaxSeqLen;

    public KVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim)
    {
        _batchSize    = batchSize;
        _numKvHeads   = numKvHeads;
        _headDim      = headDim;
        MaxSeqLen     = maxSeqLen;

        _keys   = new Tensor<float>(batchSize, numKvHeads, maxSeqLen, headDim);
        _values = new Tensor<float>(batchSize, numKvHeads, maxSeqLen, headDim);
        CurrentPosition = 0;
    }

    public int Length => CurrentPosition;
    public bool IsFull => CurrentPosition >= MaxSeqLen;

    public void Reset() => CurrentPosition = 0;

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

        CurrentPosition += seqLen;
    }

    public void Dispose()
    {
        _keys.Dispose();
        _values.Dispose();
    }
}