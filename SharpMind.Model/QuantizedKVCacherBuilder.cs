namespace SharpMind.Model;

public class QuantizedKVCacherBuilder : IKVCacheBuilder
{
    public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim)
        => new QuantizedKVCache(batchSize, numKvHeads, maxSeqLen, headDim);
}
