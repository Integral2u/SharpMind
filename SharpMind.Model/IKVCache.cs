using SharpMind.Core.Tensors;

namespace SharpMind.Model;

public interface IKVCache : IDisposable
{
    void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim);
    void Reset();
    void TrimToLast(int keep);
    int Length { get; }
    int MaxSeqLen { get; }
    bool IsFull { get; }
}
