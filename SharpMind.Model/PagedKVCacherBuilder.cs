namespace SharpMind.Model
{
    public class PagedKVCacherBuilder : IKVCacheBuilder
    {
        public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim) => new PagedKVCacheLayer(batchSize, numKvHeads, maxSeqLen, headDim);
    }
}
