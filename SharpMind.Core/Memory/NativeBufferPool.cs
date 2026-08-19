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
            // CAS pooled-marker -1 → 1: only one caller wins the race.
            // CompareExchange returns the old value, which must have been the
            // -1 pooled marker for the transition to have happened.
            if (Interlocked.CompareExchange(ref buffer._refCount, 1, -1) == -1)
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
            bool underCaps = reservedCount <= NativeBufferPoolConfig.MaxBuffersPerBucket
                && NativeBufferPoolConfig.TotalMemoryUsed + byteSize <= maxBytes;

            // Take the pooled-marker transition atomically. If a concurrent
            // AddRef (a view being created over this buffer) landed between
            // Dispose() reaching zero and here, this buffer is genuinely owned
            // by that view — TryMarkPooled loses and the buffer must stay alive
            // for it, not be pooled or freed. Its eventual Dispose drives it
            // through Return again.
            if (underCaps && buffer.TryMarkPooled())
            {
                bucket.Stack.Push(buffer);
                Interlocked.Add(ref bucket.Memory, byteSize);
                return; // Count stays; it represents this pooled buffer.
            }

            // Reservation didn't pan out — release the slot we provisionally
            // claimed so it doesn't stay permanently lost.
            Interlocked.Decrement(ref bucket.Count);

            if (buffer.TryMarkPooled())
            {
                // Nobody raced us: we hold the 0 → -1 transition, so no one is
                // using the buffer and the memory can be returned.
                buffer.Free();
                NativeBufferPoolConfig.OnFree(byteSize);
            }
            // else: a concurrent AddRef took the buffer; leave it alive.
        }
        else
        {
            // Outside the poolable size range: free unconditionally.
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
