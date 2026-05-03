using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Inference;

/// <summary>
/// Per-layer key/value cache for incremental (autoregressive) decoding.
///
/// Without a KV-cache, generating each new token requires a full forward pass
/// over the entire sequence history — O(SeqLen²) total cost. With the cache,
/// only the new token's Q attends to all cached K/V — O(SeqLen) per step.
///
/// Layout: each layer stores K and V as growing [CacheLen, NumKvHeads, HeadDim]
/// tensors. On each decode step the new K/V are appended before attention runs.
///
/// The cache is pre-allocated to <see cref="ModelConfig.MaxSeqLen"/> to avoid
/// reallocation during generation. <see cref="Length"/> tracks how many tokens
/// have been cached.
/// </summary>
public sealed class KvCache : IDisposable
{
    private readonly Tensor<float>[] _k;   // one per layer [MaxSeq, NumKvHeads, HeadDim]
    private readonly Tensor<float>[] _v;
    private bool _disposed;

    public KvCache(ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);

        NumLayers  = config.NumLayers;
        MaxSeqLen  = config.MaxSeqLen;
        NumKvHeads = config.NumKvHeads;
        HeadDim    = config.HeadDim;
        Length     = 0;

        _k = new Tensor<float>[NumLayers];
        _v = new Tensor<float>[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            _k[i] = new Tensor<float>(MaxSeqLen, NumKvHeads, HeadDim);
            _v[i] = new Tensor<float>(MaxSeqLen, NumKvHeads, HeadDim);
        }
    }

    // ── Properties ────────────────────────────────────────────────────────

    public int NumLayers  { get; }
    public int MaxSeqLen  { get; }
    public int NumKvHeads { get; }
    public int HeadDim    { get; }

    /// <summary>Number of tokens currently in the cache.</summary>
    public int Length { get; private set; }

    public bool IsFull => Length >= MaxSeqLen;

    // ── Read / Write ──────────────────────────────────────────────────────

    /// <summary>
    /// Appends new K and V slices for <paramref name="layer"/> at the current cache position.
    /// K and V must be [SeqLen, NumKvHeads, HeadDim] — the tokens generated in this step.
    /// </summary>
    public void Append(int layer, Tensor<float> k, Tensor<float> v)
    {
        ThrowIfDisposed();
        ValidateLayer(layer);

        int newTokens = k.Shape[0];
        if (Length + newTokens > MaxSeqLen)
            throw new InvalidOperationException(
                $"KV cache overflow: {Length} + {newTokens} > MaxSeqLen {MaxSeqLen}. " +
                "Truncate the context or increase MaxSeqLen.");

        int stride = NumKvHeads * HeadDim;
        k.Data.CopyTo(_k[layer].Data.Slice(Length * stride, newTokens * stride));
        v.Data.CopyTo(_v[layer].Data.Slice(Length * stride, newTokens * stride));

        // Only advance Length on the first layer — all layers get the same tokens
        if (layer == 0) Length += newTokens;
    }

    /// <summary>
    /// Returns a span view over the cached K values for <paramref name="layer"/>,
    /// up to <see cref="Length"/> tokens. Shape: [Length, NumKvHeads, HeadDim].
    /// </summary>
    public ReadOnlySpan<float> GetK(int layer)
    {
        ThrowIfDisposed();
        ValidateLayer(layer);
        return _k[layer].Data[..(Length * NumKvHeads * HeadDim)];
    }

    public ReadOnlySpan<float> GetV(int layer)
    {
        ThrowIfDisposed();
        ValidateLayer(layer);
        return _v[layer].Data[..(Length * NumKvHeads * HeadDim)];
    }

    /// <summary>
    /// Resets the cache to empty without reallocating.
    /// Call between independent generation requests.
    /// </summary>
    public void Reset()
    {
        ThrowIfDisposed();
        Length = 0;
    }

    /// <summary>
    /// Trims the cache to the last <paramref name="keepTokens"/> tokens.
    /// Used to implement a sliding window when <see cref="IsFull"/>.
    /// </summary>
    public void TrimToLast(int keepTokens)
    {
        ThrowIfDisposed();
        if (keepTokens >= Length) return;

        int drop   = Length - keepTokens;
        int stride = NumKvHeads * HeadDim;

        for (int layer = 0; layer < NumLayers; layer++)
        {
            var kData = _k[layer].Data;
            var vData = _v[layer].Data;
            kData.Slice(drop * stride, keepTokens * stride).CopyTo(kData);
            vData.Slice(drop * stride, keepTokens * stride).CopyTo(vData);
        }

        Length = keepTokens;
    }

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    private void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (!disposing) return;
        foreach (var t in _k) t.Dispose();
        foreach (var t in _v) t.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(KvCache));

    private void ValidateLayer(int layer)
    {
        if ((uint)layer >= (uint)NumLayers)
            throw new ArgumentOutOfRangeException(nameof(layer),
                $"Layer {layer} is out of range [0, {NumLayers}).");
    }
}
