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
        public int Count = 0;
        public long Memory = 0;
        public readonly object Lock = new();
    }

    private static readonly ConcurrentDictionary<int, Bucket> _buckets = new();

    public static unsafe NativeBuffer<T> Rent(int length)
    {
        if (length <= 0) length = 1;
        int bucketSize = GetBucket(length);
        
        if (_buckets.TryGetValue(bucketSize, out var bucket))
        {
            lock (bucket.Lock)
            {
                if (bucket.Stack.TryPop(out var rented))
                {
                    bucket.Count--;
                    bucket.Memory -= (long)bucketSize * sizeof(T);
                    
                    // Verify if it's already cleared? No, it should be clean.
                    return rented;
                }
            }
        }

        var buffer = new NativeBuffer<T>(bucketSize);
        NativeBufferPoolConfig.OnAllocate(bucketSize * sizeof(T));
        return buffer;
    }

    public static unsafe void Return(NativeBuffer<T> buffer)
    {
        if (buffer is null) return;
        buffer._refCount = 1;
        int bucketSize = buffer.Length; // Use actual length
        int byteSize = bucketSize * sizeof(T);
        
        var bucket = _buckets.GetOrAdd(bucketSize, _ => new Bucket());
        
        long maxMem = NativeBufferPoolConfig.MaxTotalMemoryMB * 1024 * 1024;
        int maxPerBucket = NativeBufferPoolConfig.MaxBuffersPerBucket;

        lock (bucket.Lock)
        {
            if (bucket.Count < maxPerBucket) // && (bucket.Memory + byteSize <= maxMem / 16))
            {
                NativeMemory.Clear(buffer.Ptr, (nuint)byteSize); // Clear on return
                bucket.Stack.Push(buffer);
                bucket.Count++;
                bucket.Memory += byteSize;
                return;
            }
        }

        NativeBufferPoolConfig.OnFree(byteSize);
        buffer.Free();
    }

    public static void Clear()
    {
        foreach (var bucket in _buckets.Values)
        {
            lock(bucket.Lock)
            {
                while (bucket.Stack.TryPop(out var buf))
                {
                    buf.Free();
                }
                bucket.Count = 0;
                bucket.Memory = 0;
            }
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