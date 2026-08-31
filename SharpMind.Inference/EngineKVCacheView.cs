using SharpMind.Core.Tensors;
using SharpMind.Model;

namespace SharpMind.Inference;

/// <summary>
/// Satisfies <see cref="IKVCache"/> for callers that iterate <c>IGenerator{T}.Caches</c>
/// (session snapshotting), without an engine actually exposing per-layer storage the way
/// CPU caches do. Every operation forwards to the whole-cache methods on the owning
/// <see cref="IInferenceEngine"/> — safe because the engine's cache is one unit even
/// though CPU code models it as one <see cref="IKVCache"/> per layer, so calling e.g.
/// <see cref="Reset"/> once per layer (as <c>StandardGenerator.ResetCache</c> does) just
/// calls <see cref="IInferenceEngine.ResetCache"/> a harmless extra few times.
///
/// <see cref="GetKeyPtr"/>/<see cref="GetValuePtr"/> throw: they exist for CPU attention
/// kernels reading host memory directly, and a GPU/TPU engine's cache has no host pointer
/// to give out. Nothing outside those CPU kernels calls them, and this view's own engine
/// never does either — it computes attention on-device.
/// </summary>
internal sealed class EngineKVCacheView(IInferenceEngine engine) : IKVCache
{
    /// <summary>One view per model layer, all backed by the same engine.</summary>
    public static IReadOnlyList<IKVCache> ForEngine(IInferenceEngine engine, int numLayers)
    {
        var views = new IKVCache[numLayers];
        for (int i = 0; i < numLayers; i++) views[i] = new EngineKVCacheView(engine);
        return views;
    }

    public int Length => engine.CachedLength;
    public int MaxSeqLen => engine.MaxCacheLength;
    public bool IsFull => engine.IsCacheFull;
    public bool IsContiguous => true;

    public void Reset() => engine.ResetCache();
    public void TrimToLast(int keep) => engine.TrimToLast(keep);
    public void Truncate(int length) => engine.TruncateCache(length);

    public void Update(Tensor<float> k, Tensor<float> v, int numKvHeads, int headDim) =>
        throw new NotSupportedException($"{nameof(EngineKVCacheView)} is written by {nameof(IInferenceEngine)}.Prefill/DecodeStep internally, not by a caller.");

    public unsafe float* GetKeyPtr(int batchIdx, int position, int kvHead) =>
        throw new NotSupportedException($"{nameof(EngineKVCacheView)} has no host pointer — its cache lives on the accelerator.");
    public unsafe float* GetValuePtr(int batchIdx, int position, int kvHead) =>
        throw new NotSupportedException($"{nameof(EngineKVCacheView)} has no host pointer — its cache lives on the accelerator.");

    // Deliberately NOT implemented as "call engine.ExportCache() once per layer": that would
    // recompute the whole-engine snapshot (and its PromptHash, from a prompt this per-layer
    // call doesn't have) once per layer for no reason, and get the hash wrong every time.
    // ExportCache/ImportCache are inherently whole-engine operations — a caller that wants a
    // session snapshot must go through IInferenceEngine directly, not iterate Caches. See
    // ChatSession's snapshot path, which should special-case an EngineGenerator the same way
    // it special-cases whatever ties CacheTokens to a prompt hash today, rather than looping
    // per-IKVCache.
    public object? Snapshot() => throw NotSupportedForEngineCache();
    public void Restore(object? snapshot) => throw NotSupportedForEngineCache();
    public byte[]? SnapshotBytes() => throw NotSupportedForEngineCache();
    public void RestoreBytes(byte[]? data) => throw NotSupportedForEngineCache();

    private static NotSupportedException NotSupportedForEngineCache() => new(
        $"{nameof(EngineKVCacheView)} cannot snapshot one layer at a time — call " +
        $"{nameof(IInferenceEngine)}.{nameof(IInferenceEngine.ExportCache)}/{nameof(IInferenceEngine.ImportCache)} " +
        "on the engine directly for the whole session.");

    public void Dispose() { /* the engine owns the cache; nothing to dispose per view */ }
}
