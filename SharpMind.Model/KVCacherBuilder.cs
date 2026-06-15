namespace SharpMind.Model;

public class KVCacherBuilder : IKVCacheBuilder
{
    public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim) => new KVCache(batchSize, numKvHeads, maxSeqLen, headDim);
}

