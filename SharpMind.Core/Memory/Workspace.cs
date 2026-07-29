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
    public Tensor<T> Rent<T>(ReadOnlySpan<int> shape) where T : unmanaged, INumber<T>
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
    public void Reset() => _offset = 0;

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
    /// Per-layer peak (gated FFN, non-contiguous KV temps, Q/K norms):
    ///   norm1(1H) + Q(1H) + K(1kvDim) + V(1kvDim) + attnOut(1H)
    ///   + tempK(1numH·headDim-per-seqLen) + tempV(same)
    ///   + Wo(1H) + postAttnNorm(1H)
    ///   + norm2(1H) + fused(2ffnDim) + gated(ffnDim) + down(H)
    ///   + postFfnNorm(1H) + qNorm(1H) + kNorm(1kvDim)
    ///   = up to ~12H + 3×ffnDim per seqLen-token, plus ~2×numH·headDim for temps
    ///
    /// The LM head output is a fixed [Batch, VocabSize] tensor (not per-token),
    /// allocated once per forward pass independent of sequence length.
    ///
    /// The prefill length used for sizing is capped to keep the workspace under
    /// ~1 GiB. The workspace is primarily sized for the decode hot loop (1 token);
    /// the prefill path for long prompts is handled by the generator.
    /// </summary>
    public static long CalculateRequiredSize(long hiddenDim, long ffnDim, long vocabSize, int numLayers, int maxSeqLen)
    {
        long bytesPerFloat = 4;
        long hidden = hiddenDim;
        long ffn = ffnDim;

        // Per-layer, per-token estimate (attention + q/k norms + gated FFN +
        // post-attn/post-ffn norms + non-contiguous KV temp overhead).
        // 14H covers: norm1 + Q + K(reserve H for MHA) + V + attnOut +
        //   tempK + tempV + Wo + postAttnNorm + norm2 + down + postFfnNorm +
        //   qNorm + kNorm
        // 4×ffn covers: fused(2×ffn) + gated(ffn) + headroom(1×ffn).
        long perLayer = 14 * hidden + 4 * ffn;

        // Per-token floats — all layers accumulate (bump allocator never reset mid-pass).
        // Embedding input (1H) + layers + final norm (1H).
        long perTokenFloats = hidden + numLayers * perLayer + hidden;

        // Fixed allocation — LM head / logits output [Batch, VocabSize], does NOT
        // scale with sequence length (ForwardLastLogits only projects the last token).
        long fixedFloats = vocabSize;

        // Cap prefill length to keep workspace under ~1 GiB for most models.
        int effectivePrefillLen = Math.Min(maxSeqLen, 128);
        long total = (perTokenFloats * effectivePrefillLen + fixedFloats) * bytesPerFloat;

        // Minimum 100MB to cover small-batch / short-prompt overheads.
        return Math.Max(100 * 1024 * 1024, total);
    }
}
