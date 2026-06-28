using SharpMind.Core.Tensors;
using SharpMind.Model.Format;

namespace SharpMind.Model;

public interface IKVCache : IDisposable
{
    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim);
    public void Reset();
    public void TrimToLast(int keep);
    public void Truncate(int length);
    public int Length { get; }
    public int MaxSeqLen { get; }
    public bool IsFull { get; }
    public bool IsContiguous { get; }
    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead);
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead);
    public object? Snapshot();
    public void Restore(object? snapshot);

    /// <summary>The quantization format used by quantized caches (default Q8_0).</summary>
    public GgufDtype QuantKind => GgufDtype.Q8_0;
    /// <summary>True if the cache stores quantized data.</summary>
    public bool IsQuantized => false;
    /// <summary>Returns a pointer to quantized key data at (batchIdx, position, kvHead). Only valid when IsQuantized is true.</summary>
    public unsafe byte* GetQuantizedKeyPtr(int batchIdx, int position, int kvHead) => null;
    /// <summary>Returns a pointer to quantized value data at (batchIdx, position, kvHead). Only valid when IsQuantized is true.</summary>
    public unsafe byte* GetQuantizedValuePtr(int batchIdx, int position, int kvHead) => null;
}
