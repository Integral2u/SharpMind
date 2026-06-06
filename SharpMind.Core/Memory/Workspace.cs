using System.Numerics;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Memory;

/// <summary>
/// A pre-allocated memory workspace used to eliminate GC pressure during the inference loop.
/// Provides a way to "rent" slices of a large contiguous buffer as Tensors without allocating new memory.
/// </summary>
public sealed class Workspace : IDisposable
{
    private readonly NativeBuffer<byte> _buffer;
    private long _offset;
    private readonly long _capacity;

    public Workspace(long capacityBytes)
    {
        _capacity = capacityBytes;
        _buffer = new NativeBuffer<byte>((int)capacityBytes);
        _offset = 0;
    }

    /// <summary>
    /// Returns a Tensor that views a slice of the workspace.
    /// Note: This Tensor does NOT own the memory and should NOT be disposed independently.
    /// </summary>
    public unsafe Tensor<T> Rent<T>(ReadOnlySpan<int> shape) where T : unmanaged, INumber<T>
    {
        long size = 1;
        foreach (var d in shape) size *= d;
        long bytes = size * sizeof(T);

        // Align offset to 32 bytes (AVX2 requirement)
        _offset = (_offset + 31) & ~31;

        if (_offset + bytes > _capacity)
            throw new OutOfMemoryException($"Workspace capacity exceeded. Requested {bytes} bytes, but only {_capacity - _offset} remain.");

        T* ptr = (T*)((byte*)_buffer.Ptr + _offset);
        _offset += bytes;

        // We use a special constructor for Tensor that takes an existing pointer and does not own the memory.
        return new Tensor<T>(ptr, new TensorShape(shape), ownsMemory: false);
    }

    /// <summary>
    /// Resets the offset to 0, effectively "freeing" all rented tensors for the next forward pass.
    /// </summary>
    public void Reset()
    {
        _offset = 0;
    }

    public long UsedBytes => _offset;
    public long CapacityBytes => _capacity;
    public float UsagePercentage => (float)_offset / _capacity;

    public void Dispose()
    {
        _buffer.Dispose();
    }

    /// <summary>
    /// Estimates the required workspace size based on model configuration and maximum prefill length.
    /// </summary>
    public static long CalculateRequiredSize(long hiddenDim,long ffnDim, long vocabSize,int numLayers, int maxSeqLen)
    {
        long bytesPerFloat = 4;
        long hidden = hiddenDim;
        long ffn = ffnDim;
        long vocab = vocabSize;

        // Per-token allocations across the whole model:
        // Embedding + (NumLayers * (Attention(6*hidden) + FFN(2*ffn))) + Norm
        long perToken = (hidden + numLayers * (6 * hidden + 2 * ffn) + hidden) * bytesPerFloat;

        long prefillMemory = perToken * maxSeqLen;
        long logitsMemory = vocab * bytesPerFloat;

        // Minimum 100MB to cover various overheads and small batch sizes.
        return Math.Max(100 * 1024 * 1024, prefillMemory + logitsMemory);
    }
}
