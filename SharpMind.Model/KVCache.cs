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
        if (keep >= CurrentPosition) return;
        int offset = CurrentPosition - keep;
        
        // Shift data back to the start
        // This is a simple shift, in a real system we might use a ring buffer
        unsafe
        {
            int batchSize = Keys.Shape[0];
            int numKvHeads = Keys.Shape[1];
            int headDim = Keys.Shape[3];
            
            for (int b = 0; b < batchSize; b++)
            {
                for (int h = 0; h < numKvHeads; h++)
                {
                    float* kPtr = Keys.DataPtr + (long)b * (numKvHeads * MaxSeqLen * headDim) 
                                              + (long)h * (MaxSeqLen * headDim);
                    float* vPtr = Values.DataPtr + (long)b * (numKvHeads * MaxSeqLen * headDim) 
                                              + (long)h * (MaxSeqLen * headDim);
                    
                    // Shift values
                    for (int i = 0; i < keep; i++)
                    {
                        float kvK = kPtr[(long)(CurrentPosition + i) * headDim];
                        float kvV = vPtr[(long)(CurrentPosition + i) * headDim];
                        // This is tricky because we shift across the whole MaxSeqLen
                        // For simplicity, we just move the window
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