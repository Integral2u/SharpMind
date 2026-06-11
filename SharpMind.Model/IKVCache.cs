using SharpMind.Core.Tensors;

namespace SharpMind.Model;

public interface IKVCache : IDisposable
{
    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim);
    public void Reset();
    public void TrimToLast(int keep);
    public int Length { get; }
    public int MaxSeqLen { get; }
    public bool IsFull { get; }
    public int AllocatedCapacity { get; }
    public bool IsContiguous { get; }
    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead);
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead);
    public object? Snapshot();
    public void Restore(object? snapshot);
}
