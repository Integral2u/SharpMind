using System.Buffers;
using System.Numerics;
using SharpMind.Core.Tensors;

namespace SharpMind.Core.Memory;

/// <summary>
/// Central routing layer for all memory allocations.
/// Routes ArrayPool, NativeBufferPool, and Workspace calls through a single point.
/// Returns BigArray&lt;T&gt;/BigWorkspace when &gt; int.MaxValue is needed.
/// </summary>
public static class MemoryHelpers
{
    public const int MaxArrayPoolLength = int.MaxValue / 2; // ~1 billion elements

    // For large arrays, ArrayPool power-of-2 bucketing over-allocates by up to
    // 2x (e.g. 155M floats → 268M bucket = 1 GB vs 594 MB needed).  Direct
    // allocation avoids OOM on models with large vocab embeddings (Bonsai-8B).
    private const int DirectAllocThreshold = 8 * 1024 * 1024; // 8 M elements

    /// <summary>
    /// Rents an int-sized array.  For small arrays uses <see cref="ArrayPool{T}.Shared"/>;
    /// for large arrays allocates directly to avoid power-of-2 over-allocation.
    /// </summary>
    public static T[] RentArray<T>(int length) where T : unmanaged
    {
        if (length > DirectAllocThreshold)
            return new T[length];
        return ArrayPool<T>.Shared.Rent(length);
    }

    /// <summary>
    /// Returns a rented array.  Arrays above the direct-allocation threshold
    /// are left for the GC; only pool-rented arrays are returned to the pool.
    /// </summary>
    public static void ReturnArray<T>(T[] array) where T : unmanaged
    {
        if (array is not null && array.Length <= DirectAllocThreshold)
            ArrayPool<T>.Shared.Return(array);
    }

    /// <summary>
    /// Creates a BigArray or standard int-sized array depending on the requested count.
    /// Caller must dispose the returned token to return the array to the pool.
    /// Standard case: intArray is set, bigArray is null.
    /// Oversized case: bigArray is set, intArray is null.
    /// </summary>
    public static IDisposable Rent<T>(long count, out T[]? intArray, out BigArray<T>? bigArray) where T : unmanaged
    {
        if (count <= MaxArrayPoolLength)
        {
            int intCount = (int)count;
            intArray = ArrayPool<T>.Shared.Rent(intCount);
            bigArray = null;
            return new ArrayPoolToken<T>(intArray);
        }
        else
        {
            intArray = null;
            bigArray = new BigArray<T>(count);
            return new BigArrayToken<T>(bigArray);
        }
    }

    /// <summary>
    /// Creates a workspace appropriate for the given capacity.
    /// Returns IWorkspace (Workspace or BigWorkspace).
    /// </summary>
    public static IWorkspace CreateWorkspace(long capacityBytes)
    {
        if (capacityBytes <= int.MaxValue)
            return new Workspace(capacityBytes);
        else
            return new BigWorkspace(capacityBytes);
    }

    /// <summary>
    /// Rents a NativeBuffer&lt;T&gt; through the routing layer.
    /// </summary>
    public static NativeBuffer<T> RentBuffer<T>(int length) where T : unmanaged
    {
        return NativeBufferPool<T>.Rent(length);
    }

    /// <summary>
    /// Returns a NativeBuffer&lt;T&gt; through the routing layer.
    /// </summary>
    public static void ReturnBuffer<T>(NativeBuffer<T> buffer) where T : unmanaged
    {
        NativeBufferPool<T>.Return(buffer);
    }

    private sealed class ArrayPoolToken<T> : IDisposable where T : unmanaged
    {
        private readonly T[] _array;
        public ArrayPoolToken(T[] array) => _array = array;
        public void Dispose() => ArrayPool<T>.Shared.Return(_array);
    }

    private sealed class BigArrayToken<T> : IDisposable where T : unmanaged
    {
        private readonly BigArray<T> _bigArray;
        public BigArrayToken(BigArray<T> bigArray) => _bigArray = bigArray;
        public void Dispose() => _bigArray.Dispose();
    }
}
