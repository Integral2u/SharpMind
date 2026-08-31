using SharpMind.Inference.Chat;

namespace SharpMind.Inference;

/// <summary>
/// One inference stream's compute: prefill and single-token decode, with the KV cache
/// resident wherever the engine wants (CPU tensors for the default path, device memory
/// for an accelerator). Mirrors <c>ITrainingEngine</c>'s split: the engine owns the
/// numerics and the cache; a generator (e.g. <see cref="EngineGenerator{T}"/>) owns
/// sampling, streaming, stop-string matching, and token-id bookkeeping. One engine
/// instance serves one conversation — batching multiple sessions onto one device is not
/// this interface's concern.
/// </summary>
public interface IInferenceEngine : IDisposable
{
    /// <summary>
    /// What the engine actually runs on, for the UI: e.g. <c>"CPU"</c>, or a device string like
    /// <c>"[Cuda] GeForce GTX 1060, 6144 MB, cuBLAS 12.8"</c> (ILGPU <c>GpuDevice.Description</c>).
    /// Shown in the chat status sidebar so OpenCL vs. ILGPU-CUDA vs. cuBLAS vs. CPU is always visible.
    /// </summary>
    string Description { get; }

    /// <summary>Position count currently resident in the KV cache.</summary>
    int CachedLength { get; }

    /// <summary>Cache capacity in positions. <see cref="CachedLength"/> never exceeds this.</summary>
    int MaxCacheLength { get; }

    /// <summary><see cref="CachedLength"/> == <see cref="MaxCacheLength"/>.</summary>
    bool IsCacheFull { get; }

    /// <summary>
    /// Runs the model over <paramref name="tokenIds"/> starting at <see cref="CachedLength"/>,
    /// writing each position's K/V into the cache, and returns logits for the LAST token
    /// only — matching <c>Prefill.ForwardLastLogitsChunked</c>'s contract, since nothing
    /// needs per-position logits during prefill. <paramref name="onChunkProgress"/> reports
    /// the running fraction (0..1) if the engine chunks internally against a memory budget.
    /// </summary>
    /// <returns>
    /// A view over a buffer this engine owns and reuses. Valid until the NEXT call to
    /// <see cref="Prefill"/> or <see cref="DecodeStep"/> on this instance — same rental
    /// contract as <c>IWorkspace.Rent&lt;T&gt;</c>. Callers that need the values past that
    /// point must copy them.
    /// </returns>
    ReadOnlyMemory<float> Prefill(ReadOnlySpan<int> tokenIds, Action<double>? onChunkProgress = null,
        CancellationToken cancellationToken = default);

    /// <summary>
    /// Runs one autoregressive step for <paramref name="tokenId"/> at position
    /// <see cref="CachedLength"/>, appends it to the cache, and returns that position's
    /// logits under the same buffer-reuse contract as <see cref="Prefill"/>.
    /// </summary>
    ReadOnlyMemory<float> DecodeStep(int tokenId, CancellationToken cancellationToken = default);

    /// <summary>Drops the cache back to <paramref name="length"/>. See <c>IGenerator{T}.TruncateCache</c>.</summary>
    void TruncateCache(int length);

    /// <summary>Keeps only the last <paramref name="keep"/> cached positions. See <c>IKVCache.TrimToLast</c>.</summary>
    void TrimToLast(int keep);

    /// <summary>Empties the cache. Equivalent to <c>TruncateCache(0)</c>.</summary>
    void ResetCache();

    /// <summary>
    /// Copies the live cache to host memory in the same per-layer tag-1 blob format
    /// <c>IKVCache.SnapshotBytes()</c> produces, so a saved session round-trips through
    /// either a CPU or GPU engine interchangeably. Device engines pay a PCIe download
    /// here — expected on session save, not per token.
    /// </summary>
    KVCacheSnapshot ExportCache(int[] promptTokenIds);

    /// <summary>Restores a previously exported cache. Throws if the layer count/shape doesn't match this engine's model.</summary>
    void ImportCache(KVCacheSnapshot snapshot);
}
