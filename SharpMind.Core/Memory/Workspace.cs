using System.Numerics;
using System.Runtime.InteropServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Memory;

/// <summary>
/// A pre-allocated memory workspace used to eliminate GC pressure during the inference loop.
/// Provides a way to "rent" slices of a large contiguous buffer as Tensors without allocating new memory.
///
/// Uses a linear bump allocator: rents advance an offset, and <see cref="Reset"/> rewinds it.
/// Tensors obtained via <see cref="Rent{T}"/> do not own memory and must not be disposed independently.
/// </summary>
public sealed unsafe class Workspace : IDisposable
{
    private byte* _buffer;
    private long _offset;
    private readonly long _capacity;

    public Workspace(long capacityBytes)
    {
        _capacity = capacityBytes;
        unsafe
        {
            _buffer = (byte*)NativeMemory.AlignedAlloc((nuint)capacityBytes, 32);
            if (_buffer is null)
                throw new OutOfMemoryException("Workspace: aligned allocation failed.");
            NativeMemory.Clear(_buffer, (nuint)capacityBytes);
        }
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

        T* ptr = (T*)(_buffer + _offset);
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
        unsafe
        {
            if (_buffer != null)
            {
                NativeMemory.AlignedFree(_buffer);
                _buffer = null;
            }
        }
    }

    /// <summary>
    /// Estimates the required workspace size based on model configuration and maximum prefill length.
    ///
    /// The formula accounts for all intermediate tensors allocated by a single forward pass
    /// through all layers (the bump allocator does not free between layers):
    ///   Embedding + NumLayers * (Norm + Attention + FFN) + FinalNorm + LM head
    ///
    /// Attention counts (non-contiguous cache, worst case):
    ///   Q + K + V + output + tempK + tempV + Wo + norm1 = 9 * hidden
    ///
    /// Gated FFN counts:
    ///   gate-up + gate-split + up-split + gate-act + down + norm2 = 2*hidden + 5*ffnDim
    ///
    /// The prefill length used for sizing is capped to keep the workspace under 2 GiB
    /// and to avoid OS commit-pressure issues. Full prefill of very long sequences
    /// should fall back to direct allocations (handled by the generator).
    /// </summary>
    public static long CalculateRequiredSize(long hiddenDim, long ffnDim, long vocabSize, int numLayers, int maxSeqLen)
    {
        long bytesPerFloat = 4;
        long hidden = hiddenDim;
        long ffn = ffnDim;
        long vocab = vocabSize;

        // Per-token floats for the entire forward pass — all layers accumulate
        // because the bump allocator is never reset mid-pass.
        // Estimate: norm(1H) + attn(8H+2kvDim) + gated-ffn(2H+5ffn) per layer
        // plus embedding(1H) + finalNorm(1H) + LMhead(1V).
        long perLayer = 11 * hidden + 7 * ffn;  // generous upper bound
        long perTokenFloats = hidden + numLayers * perLayer + hidden + vocab;

        // Cap prefill length to keep the workspace under ~2 GiB. The workspace
        // is primarily sized for the decode hot loop; very long prefills will
        // fall back to direct allocation (handled by the generator).
        int effectivePrefillLen = Math.Min(maxSeqLen, 256);
        long total = perTokenFloats * bytesPerFloat * effectivePrefillLen;

        // Minimum 100MB to cover small-batch / short-prompt overheads.
        return Math.Max(100 * 1024 * 1024, total);
    }
}
