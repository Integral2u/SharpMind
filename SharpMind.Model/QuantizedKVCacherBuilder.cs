using SharpMind.Core.Quantization;

namespace SharpMind.Model;

public class QuantizedKVCacherBuilder(QuantDType quantKind) : IKVCacheBuilder
{
    public QuantDType QuantKind { get; } = quantKind;

    public QuantizedKVCacherBuilder() : this(QuantDType.Q8_0)
    {
    }

    public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim)
        => new QuantizedKVCache(batchSize, numKvHeads, maxSeqLen, headDim, QuantKind);
}
