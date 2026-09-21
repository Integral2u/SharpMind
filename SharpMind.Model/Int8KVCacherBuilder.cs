namespace SharpMind.Model;

public class Int8KVCacherBuilder : IKVCacheBuilder
{
    public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim) => new Int8KVCache(batchSize, numKvHeads, maxSeqLen, headDim);
}
