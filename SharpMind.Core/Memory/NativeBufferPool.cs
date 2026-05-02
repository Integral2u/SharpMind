using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace SharpMind.Core.Memory;

/// <summary>
/// A simple thread-safe pool for <see cref="NativeBuffer{T}"/> allocations.
/// Reduces GC pressure and native allocation overhead for temporary tensors.
/// </summary>
public static class NativeBufferPool<T> where T : unmanaged
{
    private static readonly ConcurrentDictionary<int, ConcurrentStack<NativeBuffer<T>>> _pools = new();

    /// <summary>Rents a buffer of at least the requested length.</summary>
    public static NativeBuffer<T> Rent(int length)
    {
        // We bucket by power-of-two to increase reuse rates
        int bucket = GetBucket(length);
        NativeBuffer<T> buffer;

        if (_pools.TryGetValue(bucket, out var stack) && stack.TryPop(out var rented))
        {
            buffer = rented;
        }
        else
        {
            buffer = new NativeBuffer<T>(bucket);
        }

        // Ensure the buffer is zero-initialised before returning
        unsafe
        {
            NativeMemory.Clear(buffer.Ptr, (nuint)(bucket * sizeof(T)));
        }

        return buffer;
    }

    /// <summary>Returns a buffer to the pool for future reuse.</summary>
    public static void Return(NativeBuffer<T> buffer)
    {
        int bucket = GetBucket(buffer.Length);
        var stack = _pools.GetOrAdd(bucket, _ => new ConcurrentStack<NativeBuffer<T>>());
        stack.Push(buffer);
    }

    public static int GetBucket(int length)
    {
        if (length <= 0) return 0;
        int bucket = 1;
        while (bucket < length) bucket <<= 1;
        return bucket;
    }
}