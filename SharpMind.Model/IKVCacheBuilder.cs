namespace SharpMind.Model
{
    public interface IKVCacheBuilder
    {
        public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim);

        /// <summary>
        /// MLA caches one <paramref name="latentDim"/>-wide latent per position
        /// instead of a key and a value per head. Builders that do not specialise
        /// it keep the expanded cache — correct, just larger — so a builder that
        /// knows nothing about MLA still runs the model.
        /// </summary>
        public IKVCache CreateMlaKvCache(int batchSize, int maxSeqLen, int latentDim) =>
            CreateKVCache(batchSize, 1, maxSeqLen, latentDim);
    }
}
