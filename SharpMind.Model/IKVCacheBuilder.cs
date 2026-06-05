using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Threading.Tasks;

namespace SharpMind.Model
{
    public interface IKVCacheBuilder
    {
        public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim);
    }
    public class KVCacherBuilder : IKVCacheBuilder
    {
        public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim) => new KVCache(batchSize, numKvHeads, maxSeqLen, headDim);
    }
    public class PagedKVCacherBuilder : IKVCacheBuilder
    {
        public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim) => new PagedKVCacheLayer(batchSize, numKvHeads, maxSeqLen, headDim);
    }
}
