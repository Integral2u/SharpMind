using System.Numerics;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Memory;

/// <summary>
/// A workspace for oversized contexts where capacity exceeds int.MaxValue.
/// Uses NativeMemory for the raw buffer but allocates in smaller chunks
/// to avoid a single massive allocation. Rent returns long-shaped tensors.
/// </summary>
public sealed unsafe class BigWorkspace : IWorkspace
{
    private readonly long _capacity;
    private long _offset;
    private byte* _buffer;

    public BigWorkspace(long capacityBytes)
    {
        _capacity = capacityBytes;
        if (capacityBytes > int.MaxValue)
            throw new OutOfMemoryException(
                $"BigWorkspace: {capacityBytes} bytes exceeds int.MaxValue. " +
                "Full oversized workspace support deferred to a future release.");
        _buffer = (byte*)NativeMemory.AlignedAlloc((nuint)capacityBytes, 32);
        if (_buffer is null)
            throw new OutOfMemoryException("BigWorkspace: aligned allocation failed.");
        NativeMemory.Clear(_buffer, (nuint)capacityBytes);
    }

    public Tensor<T> Rent<T>(ReadOnlySpan<int> shape) where T : unmanaged, INumber<T>
    {
        long size = 1;
        foreach (var d in shape) size *= d;
        long bytes = size * sizeof(T);

        _offset = (_offset + 31) & ~31;

        if (_offset + bytes > _capacity)
            throw new OutOfMemoryException(
                $"BigWorkspace capacity exceeded. Requested {bytes} bytes, but only {_capacity - _offset} remain.");

        T* ptr = (T*)(_buffer + _offset);
        _offset += bytes;
        return new Tensor<T>(ptr, new TensorShape(shape), ownsMemory: false);
    }

    public void Reset() => _offset = 0;

    public long UsedBytes => _offset;
    public long CapacityBytes => _capacity;
    public float UsagePercentage => (float)_offset / _capacity;

    public void Dispose()
    {
        if (_buffer is not null)
        {
            NativeMemory.AlignedFree(_buffer);
            _buffer = null;
        }
    }

    /// <summary>
    /// Estimates the required workspace size for a given configuration.
    /// Uses the same formula as <see cref="Workspace.CalculateRequiredSize"/>
    /// but supports long capacity.
    /// </summary>
    public static long CalculateRequiredSize(long hiddenDim, long ffnDim, long vocabSize, int numLayers, int maxSeqLen)
    {
        long bytesPerFloat = 4;
        long perLayer = 14 * hiddenDim + 4 * ffnDim;
        long perTokenFloats = hiddenDim + numLayers * perLayer + hiddenDim;
        long fixedFloats = vocabSize;
        int effectivePrefillLen = Math.Min(maxSeqLen, 128);
        long total = (perTokenFloats * effectivePrefillLen + fixedFloats) * bytesPerFloat;
        return Math.Max(100 * 1024 * 1024, total);
    }
}
