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
        public int Count;
        public long Memory;
    }

    private static readonly ConcurrentDictionary<int, Bucket> _buckets = new();

    public static unsafe NativeBuffer<T> Rent(int length)
    {
        if (length <= 0) length = 1;
        int bucketSize = GetBucket(length);

        if (_buckets.TryGetValue(bucketSize, out var bucket) && bucket.Stack.TryPop(out var buffer))
        {
            // CAS refcount 0 → 1: only one caller wins the race.
            if (Interlocked.CompareExchange(ref buffer._refCount, 1, 0) == 0)
            {
                Interlocked.Decrement(ref bucket.Count);
                long byteLen = (long)bucketSize * sizeof(T);
                Interlocked.Add(ref bucket.Memory, -byteLen);
                NativeMemory.Clear(buffer.Ptr, (nuint)byteLen);
                // Bytes were already counted in TotalMemoryUsed when originally allocated;
                // they stayed in the pool (not freed), so do NOT call OnAllocate again.
                // Dispose() suppressed this buffer's finalizer when it was last
                // returned to the pool; re-arm it now that it's leased out again,
                // so a leak on *this* lease is still caught even though the
                // buffer itself has already outlived its original allocation.
                GC.ReRegisterForFinalize(buffer);
                return buffer;
            }
        }

        var newBuffer = new NativeBuffer<T>(bucketSize);
        NativeBufferPoolConfig.OnAllocate(bucketSize * sizeof(T));
        return newBuffer;
    }

    public static unsafe void Return(NativeBuffer<T> buffer)
    {
        if (buffer is null) return;

        int bucketSize = buffer.Length;
        int byteSize = bucketSize * sizeof(T);
        long maxBytes = NativeBufferPoolConfig.MaxTotalMemoryMB * 1024 * 1024;

        bool pooled = false;
        if (bucketSize <= 1024 * 1024)
        {
            var bucket = _buckets.GetOrAdd(bucketSize, _ => new Bucket());
            // Reserve the bucket slot first, then check whether the
            // reservation landed under the cap — Count is a shared claimable
            // resource, so this needs to be atomic (see comment below).
            //
            // TotalMemoryUsed, by contrast, does NOT need the same treatment:
            // this buffer's bytes were already added to it back when the
            // buffer was first constructed (see Rent()'s cache-miss path)
            // and stay counted the whole time it exists, in use or pooled —
            // pooling it here doesn't newly allocate anything, so there's no
            // shared resource being claimed for two threads to race over.
            // A plain read is enough; it's a pressure-valve heuristic
            // ("free more aggressively as we approach the cap"), not a lock.
            int reservedCount = Interlocked.Increment(ref bucket.Count);
            if (reservedCount <= NativeBufferPoolConfig.MaxBuffersPerBucket
                && NativeBufferPoolConfig.TotalMemoryUsed + byteSize <= maxBytes)
{
                buffer.Detach();       // mark as pooled, do NOT free memory
                bucket.Stack.Push(buffer);
                Interlocked.Add(ref bucket.Memory, byteSize);
                pooled = true;
            }
            else
            {
                // Reservation didn't pan out — release the slot we
                // provisionally claimed so it doesn't stay permanently lost.
                Interlocked.Decrement(ref bucket.Count);
            }
        }

        if (!pooled)
        {
            buffer.Free();
            NativeBufferPoolConfig.OnFree(byteSize);
        }
    }

    public static unsafe void Clear()
    {
        foreach (var bucket in _buckets.Values)
        {
            while (bucket.Stack.TryPop(out var buf))
            {
                long byteLen = (long)buf.Length * sizeof(T);
                buf.Free();
                NativeBufferPoolConfig.OnFree((int)byteLen);
            }
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
