using System;
using System.Runtime.InteropServices;
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
    public unsafe Tensor<float> Rent(int[] shape)
    {
        long size = 1;
        foreach (var d in shape) size *= d;
        long bytes = size * sizeof(float);

        if (_offset + bytes > _capacity)
            throw new OutOfMemoryException($"Workspace capacity exceeded. Requested {bytes} bytes, but only {_capacity - _offset} remain.");

        float* ptr = (float*)_buffer.Ptr + (_offset / sizeof(float));
        _offset += bytes;

        // We use a special constructor for Tensor that takes an existing pointer and does not own the memory.
        return new Tensor<float>(ptr, new TensorShape(shape), ownsMemory: false);
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
}
