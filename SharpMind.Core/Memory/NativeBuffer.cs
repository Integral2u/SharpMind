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
    // constants
    public const nuint Alignment = 32; // AVX2

    // fields 
    internal T* _ptr;
    // Positive while live (1 on rent/construction, +1 per AddRef), and -1 while
    // pooled in NativeBufferPool<T> awaiting reuse. Only the pool writes -1,
    // via TryMarkPooled, so it never races a live AddRef.
    internal int _refCount = 1;

    // construction

    public NativeBuffer(int elementCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(elementCount);
        Length = elementCount;
        nuint byteLen = (nuint)elementCount * (nuint)sizeof(T);
        _ptr = (T*)NativeMemory.AlignedAlloc(byteLen, Alignment);
        if (_ptr is null) ThrowOom();
        NativeMemory.Clear(_ptr, byteLen);
    }

    // properties

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

    // ref-counting

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
            // pool actually frees the memory now or keeps it for reuse,
            // nothing further needs finalizing for *this* lease — if the
            // buffer gets rented out again later, Rent() re-arms the
            // finalizer for that next lease (see NativeBufferPool<T>.Rent).
            GC.SuppressFinalize(this);
            NativeBufferPool<T>.Return(this);
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

    /// <summary>
    /// Safety net only — Dispose()/the pool are the intended cleanup path.
    /// This exists purely to reclaim native memory if a buffer is ever
    /// dropped without going through Dispose() (e.g. an exception between
    /// allocation and a `using`, or a missed disposal on an error path).
    /// Reachability guarantees this never fires while a buffer is legitimately
    /// sitting in NativeBufferPool's stack awaiting reuse — anything the pool
    /// still references can't be unreachable, and Free() being idempotent
    /// makes this safe even if cleanup already happened through the normal path.
    /// 
    /// Suppressed from Dispose() (see above), not from here — CA1816 expects
    /// GC.SuppressFinalize to be called by Dispose() specifically. Each time
    /// a pooled buffer is handed out again by Rent(), the finalizer is
    /// re-armed via GC.ReRegisterForFinalize so this safety net covers every
    /// lease of a reused buffer, not just its first one.
    /// </summary>
    ~NativeBuffer()
    {
        Free();
    }

    /// <summary>
    /// Attempts to mark this buffer as pooled without freeing native memory.
    /// The pooled marker is -1 (all other states are live reference counts).
    /// Uses CompareExchange so a concurrent <see cref="AddRef"/> from a view
    /// being created between our <see cref="Dispose"/> hitting zero and here is
    /// not clobbered: that view genuinely owns the buffer now, so this loses
    /// the race and the pool must not take it.
    /// </summary>
    internal bool TryMarkPooled() => Interlocked.CompareExchange(ref _refCount, -1, 0) == 0;

    // span access

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

    // element access

    public ref T this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => ref _ptr[index];
    }

    // Helpers

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AssertAlive()
    {
        if (_ptr is null)
            ThrowDisposed();
    }

    private static void ThrowOom() =>
        throw new OutOfMemoryException("NativeBuffer: aligned allocation failed.");

    private static void ThrowDisposed() =>
        throw new ObjectDisposedException(nameof(NativeBuffer<>));
}
