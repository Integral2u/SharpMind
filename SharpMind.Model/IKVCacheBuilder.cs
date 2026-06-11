namespace SharpMind.Model
{
    public interface IKVCacheBuilder
    {
        public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim);
    }
}
