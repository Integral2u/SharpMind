using SharpMind.Core.Quantization;

namespace SharpMind.Model;

public class QuantizedKVCacherBuilder : IKVCacheBuilder
{
    public QuantDType QuantKind { get; }

    public QuantizedKVCacherBuilder() : this(QuantDType.Q8_0)
    {
    }

    public QuantizedKVCacherBuilder(QuantDType quantKind)
    {
        QuantKind = quantKind;
    }

    public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim)
        => new QuantizedKVCache(batchSize, numKvHeads, maxSeqLen, headDim, QuantKind);
}
