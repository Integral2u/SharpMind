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

public static class NativeBufferPoolStats
{
    private static readonly ConcurrentDictionary<int, int> _bucketCounts = new();
    public static int GetPooledCount(int bucket) => _bucketCounts.TryGetValue(bucket, out var c) ? c : 0;
    public static int GetBucketCount() => _bucketCounts.Count;
    internal static void Increment(int bucket) => _bucketCounts.AddOrUpdate(bucket, 1, (_, c) => c + 1);
    internal static void Decrement(int bucket) => _bucketCounts.AddOrUpdate(bucket, 1, (_, c) => c - 1);
}

public static class NativeBufferPool<T> where T : unmanaged
{
    private static readonly ConcurrentDictionary<int, ConcurrentStack<NativeBuffer<T>>> _pools = new();
    private static readonly ConcurrentDictionary<int, int> _bucketCounts = new();
    private static readonly ConcurrentDictionary<int, long> _bucketMemory = new();

    public static NativeBuffer<T> Rent(int length)
    {
        if (length <= 0) length = 1;
        int bucket = GetBucket(length);
        NativeBuffer<T> buffer;

        if (_pools.TryGetValue(bucket, out var stack) && stack.TryPop(out var rented))
        {
            buffer = rented;
        }
        else
        {
            buffer = new NativeBuffer<T>(bucket);
            unsafe { NativeBufferPoolConfig.OnAllocate(bucket * sizeof(T)); }
        }

        unsafe { NativeMemory.Clear(buffer.Ptr, (nuint)(bucket * sizeof(T))); }
        return buffer;
    }

    public static void Return(NativeBuffer<T> buffer)
    {
        if (buffer is null) return;
        buffer._refCount = 1;
        int bucket = GetBucket(buffer.Length);
        int byteSize;
        unsafe { byteSize = bucket * sizeof(T); }
        long maxMem = NativeBufferPoolConfig.MaxTotalMemoryMB * 1024 * 1024;
        long currentMem = _bucketMemory.GetOrAdd(bucket, 0L);

        if (currentMem + byteSize > maxMem / 16)
        {
            unsafe { NativeBufferPoolConfig.OnFree(bucket * sizeof(T)); }
            buffer.Free();
            return;
        }

        int maxPerBucket = NativeBufferPoolConfig.MaxBuffersPerBucket;
        int currentCount = _bucketCounts.GetOrAdd(bucket, 0);
        if (currentCount >= maxPerBucket)
        {
            unsafe { NativeBufferPoolConfig.OnFree(bucket * sizeof(T)); }
            buffer.Free();
            return;
        }

        var stack = _pools.GetOrAdd(bucket, _ => new ConcurrentStack<NativeBuffer<T>>());
        stack.Push(buffer);
        _bucketCounts.AddOrUpdate(bucket, 1, (_, c) => c + 1);
        _bucketMemory.AddOrUpdate(bucket, byteSize, (_, m) => m + byteSize);
    }

    public static void Clear()
    {
        foreach (var kvp in _pools)
        {
            while (kvp.Value.TryPop(out _)) { }
        }
        _pools.Clear();
        _bucketCounts.Clear();
        _bucketMemory.Clear();
    }

    public static int GetBucket(int length)
    {
        if (length <= 0) return 0;
        int bucket = 1;
        while (bucket < length) bucket <<= 1;
        return bucket;
    }
}