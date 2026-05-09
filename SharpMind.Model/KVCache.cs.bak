using SharpMind.Core.Tensors;

namespace SharpMind.Model;

/// <summary>
/// Stores cached Key and Value tensors for a single transformer layer.
/// This allows auto-regressive generation to avoid re-computing the entire sequence.
/// </summary>
public sealed class KVCache : IDisposable
{
    public Tensor<float> Keys { get; }
    public Tensor<float> Values { get; }
    public int CurrentPosition { get; private set; }
    public int MaxSeqLen { get; }

    public KVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim)
    {
        MaxSeqLen = maxSeqLen;
        Keys = new Tensor<float>(batchSize, numKvHeads, maxSeqLen, headDim);
        Values = new Tensor<float>(batchSize, numKvHeads, maxSeqLen, headDim);
        CurrentPosition = 0;
    }

    public int Length => CurrentPosition;
    public bool IsFull => CurrentPosition >= MaxSeqLen;

    public void Reset() => CurrentPosition = 0;

    public void TrimToLast(int keep)
    {
        if (keep < 0)
            throw new ArgumentOutOfRangeException(nameof(keep));
        if (keep >= CurrentPosition) return;

        int offset = CurrentPosition - keep;
        unsafe
        {
            int batchSize = Keys.Shape[0];
            int numKvHeads = Keys.Shape[1];
            int headDim = Keys.Shape[3];
            long tokenStride = (long)headDim * sizeof(float);
            
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < numKvHeads; h++)
                {
                    float* kPtr = Keys.DataPtr + (long)b * (numKvHeads * MaxSeqLen * headDim) 
                                              + (long)h * (MaxSeqLen * headDim);
                    float* vPtr = Values.DataPtr + (long)b * (numKvHeads * MaxSeqLen * headDim) 
                                              + (long)h * (MaxSeqLen * headDim);

                    // Move the retained window [offset, offset+keep) to [0, keep).
                    for (int i = 0; i < keep; i++)
                    {
                        float* srcK = kPtr + (long)(offset + i) * headDim;
                        float* dstK = kPtr + (long)i * headDim;
                        Buffer.MemoryCopy(srcK, dstK, tokenStride, tokenStride);

                        float* srcV = vPtr + (long)(offset + i) * headDim;
                        float* dstV = vPtr + (long)i * headDim;
                        Buffer.MemoryCopy(srcV, dstV, tokenStride, tokenStride);
                    }
                }
            }
        }
        CurrentPosition = keep;
    }

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim)
    {
        int batch = k.Shape[0];
        int seqLen = k.Shape[1];
        if (CurrentPosition + seqLen > MaxSeqLen)
            throw new InvalidOperationException(
                $"KVCache overflow: position {CurrentPosition} + seqLen {seqLen} exceeds capacity {MaxSeqLen}.");

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numKvHeads; h++)
                {
                    unsafe
                    {
                        float* srcPtr = k.DataPtr + (long)b * (seqLen * numKvHeads * headDim) 
                                                   + (long)s * (numKvHeads * headDim) 
                                                   + (long)h * headDim;
                        
                        float* dstPtr = Keys.DataPtr + (long)b * (numKvHeads * MaxSeqLen * headDim) 
                                                     + (long)h * (MaxSeqLen * headDim) 
                                                     + (long)(CurrentPosition + s) * headDim;

                        for (int d = 0; d < headDim; d++)
                            dstPtr[d] = srcPtr[d];

                        srcPtr = v.DataPtr + (long)b * (seqLen * numKvHeads * headDim) 
                                           + (long)s * (numKvHeads * headDim) 
                                           + (long)h * headDim;
                        
                        dstPtr = Values.DataPtr + (long)b * (numKvHeads * MaxSeqLen * headDim) 
                                                 + (long)h * (MaxSeqLen * headDim) 
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
        Keys.Dispose();
        Values.Dispose();
    }
}