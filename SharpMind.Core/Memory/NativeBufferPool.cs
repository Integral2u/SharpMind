using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace SharpMind.Core.Memory;

public static class NativeBufferPoolConfig
{
    public static int MaxBuffersPerBucket { get; set; } = 64;
    public static long MaxTotalMemoryMB { get; set; } = 512;
    public static long TotalMemoryUsed => Interlocked.Read(ref _totalMemoryUsed);
    private static long _totalMemoryUsed;
    internal static void OnAllocate(int byteSize) => Interlocked.Add(ref _totalMemoryUsed, byteSize);
    internal static void OnFree(int byteSize) => Interlocked.Add(ref _totalMemoryUsed, -byteSize);
}

public static class NativeBufferPool<T> where T : unmanaged
{
    private class Bucket
    {
        public readonly ConcurrentStack<NativeBuffer<T>> Stack = new();
        /// <summary>Approximate count — Interlocked for lock-free access.</summary>
        public int Count = 0;
        /// <summary>Approximate memory — Interlocked for lock-free access.</summary>
        public long Memory = 0;
    }

    private static readonly ConcurrentDictionary<int, Bucket> _buckets = new();

    public static unsafe NativeBuffer<T> Rent(int length)
    {
        if (length <= 0) length = 1;
        int bucketSize = GetBucket(length);

        if (_buckets.TryGetValue(bucketSize, out var bucket)
            && bucket.Stack.TryPop(out var rented))
        {
            Interlocked.Decrement(ref bucket.Count);
            Interlocked.Add(ref bucket.Memory, -(long)bucketSize * sizeof(T));
            return rented;
        }

        var buffer = new NativeBuffer<T>(bucketSize);
        NativeBufferPoolConfig.OnAllocate(bucketSize * sizeof(T));
        return buffer;
    }

    public static unsafe void Return(NativeBuffer<T> buffer)
    {
        if (buffer is null) return;
        buffer._refCount = 1;
        int bucketSize = buffer.Length;
        int byteSize = bucketSize * sizeof(T);

        var bucket = _buckets.GetOrAdd(bucketSize, _ => new Bucket());
        int maxPerBucket = NativeBufferPoolConfig.MaxBuffersPerBucket;

        int newCount = Interlocked.Increment(ref bucket.Count);
        Interlocked.Add(ref bucket.Memory, byteSize);

        if (newCount <= maxPerBucket)
        {
            NativeMemory.Clear(buffer.Ptr, (nuint)byteSize);
            bucket.Stack.Push(buffer);
        }
        else
        {
            Interlocked.Decrement(ref bucket.Count);
            Interlocked.Add(ref bucket.Memory, -byteSize);
            NativeBufferPoolConfig.OnFree(byteSize);
            buffer.Free();
        }
    }

    public static void Clear()
    {
        foreach (var bucket in _buckets.Values)
        {
            while (bucket.Stack.TryPop(out var buf))
                buf.Free();
            bucket.Count = 0;
            bucket.Memory = 0;
        }
        _buckets.Clear();
    }

    public static int GetBucket(int length)
    {
        if (length <= 0) return 0;
        int bucket = 1;
        while (bucket < length) bucket <<= 1;
        return bucket;
    }
}