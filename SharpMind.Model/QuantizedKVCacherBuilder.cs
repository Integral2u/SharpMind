using SharpMind.Model.Format;

namespace SharpMind.Model;

public class QuantizedKVCacherBuilder : IKVCacheBuilder
{
    public GgufDtype QuantKind { get; }

    public QuantizedKVCacherBuilder() : this(GgufDtype.Q8_0)
    {
    }

    public QuantizedKVCacherBuilder(GgufDtype quantKind)
    {
        QuantKind = quantKind;
    }

    public IKVCache CreateKVCache(int batchSize, int numKvHeads, int maxSeqLen, int headDim)
        => new QuantizedKVCache(batchSize, numKvHeads, maxSeqLen, headDim, QuantKind);
}
