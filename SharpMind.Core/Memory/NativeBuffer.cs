using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace SharpMind.Core.Memory;

/// <summary>
/// Owns a block of native (unmanaged), SIMD-aligned memory.
/// Reference-counted so multiple <c>Tensor&lt;T&gt;</c> views can safely share
/// the same allocation without copying. The memory is freed when the last
/// reference is released via <see cref="Dispose"/>.
/// </summary>
/// <remarks>
/// Alignment is 32 bytes (AVX2 / 256-bit lane). All allocations are
/// zero-initialised on construction.
/// </remarks>
public sealed unsafe class NativeBuffer<T> : IDisposable where T : unmanaged
{
    // ── constants ──────────────────────────────────────────────────────────
    public const nuint Alignment = 32; // AVX2

    // ── fields ─────────────────────────────────────────────────────────────
    private T* _ptr;
    private int _refCount = 1;

    // ── construction ───────────────────────────────────────────────────────

    public NativeBuffer(int elementCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(elementCount);
        Length = elementCount;
        nuint byteLen = (nuint)(elementCount * sizeof(T));
        _ptr = (T*)NativeMemory.AlignedAlloc(byteLen, Alignment);
        if (_ptr is null) ThrowOom();
        NativeMemory.Clear(_ptr, byteLen);
    }

    // ── properties ─────────────────────────────────────────────────────────

    public int Length { get; }

    /// <summary>Raw pointer. Valid only while this buffer is alive.</summary>
    public T* Ptr
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            AssertAlive();
            return _ptr;
        }
    }

    // ── ref-counting ───────────────────────────────────────────────────────

    /// <summary>
    /// Increment the reference count. Called when a view tensor borrows this buffer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void AddRef() => Interlocked.Increment(ref _refCount);

    /// <summary>
    /// Decrement the reference count. Frees native memory when it hits zero.
    /// </summary>
    public void Dispose()
    {
        if (Interlocked.Decrement(ref _refCount) == 0)
        {
            // Disable pooling for now - it causes reuse issues
            Free();
        }
    }

    internal void Free()
    {
        if (_ptr is not null)
        {
            NativeMemory.AlignedFree(_ptr);
            _ptr = null;
        }
    }

    // ── span access ────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan()
    {
        AssertAlive();
        return new Span<T>(_ptr, Length);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<T> AsSpan(int start, int length)
    {
        AssertAlive();
        return new Span<T>(_ptr + start, length);
    }

    // ── element access ─────────────────────────────────────────────────────

    public ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref _ptr[index];
    }

    // ── helpers ────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AssertAlive()
    {
        if (_ptr is null)
            ThrowDisposed();
    }

    private static void ThrowOom() =>
        throw new OutOfMemoryException("NativeBuffer: aligned allocation failed.");

    private static void ThrowDisposed() =>
        throw new ObjectDisposedException(nameof(NativeBuffer<T>));
}
