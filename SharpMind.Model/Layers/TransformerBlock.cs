using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.Model.Layers;
/// <summary>
/// A single transformer block: pre-norm → attention → residual →
///                             pre-norm → FFN → residual.
///
/// Pre-norm (norm before the sub-layer) is used by all modern LLMs.
/// Post-norm (BERT-style) is not currently supported but can be added
/// by reordering the forward pass here.
/// </summary>
public sealed class TransformerBlock : IDisposable
{
    private readonly AttentionLayer _attention;
    private readonly FfnLayer _ffn;
    private readonly NormLayer _norm1;   // pre-attention norm
    private readonly NormLayer _norm2;   // pre-FFN norm
    private readonly TensorOps _ops;
    private bool _disposed;

    private Tensor<float>? _cachedInput;
    private Tensor<float>? _cachedNormed1;
    private Tensor<float>? _cachedAttnOut;
    private Tensor<float>? _cachedHidden;
    private Tensor<float>? _cachedNormed2;
    private Tensor<float>? _cachedFfnOut;

    public TransformerBlock(
        AttentionLayer attention,
        FfnLayer ffn,
        NormLayer norm1,
        NormLayer norm2,
        TensorOps ops)
    {
        ArgumentNullException.ThrowIfNull(attention);
        ArgumentNullException.ThrowIfNull(ffn);
        ArgumentNullException.ThrowIfNull(norm1);
        ArgumentNullException.ThrowIfNull(norm2);
        ArgumentNullException.ThrowIfNull(ops);

        _attention = attention;
        _ffn = ffn;
        _norm1 = norm1;
        _norm2 = norm2;
        _ops = ops;
    }

    // ── Forward ───────────────────────────────────────────────────────────

    /// <summary>
    /// Single block forward pass with residual connections.
    /// Input/output: [Batch, SeqLen, HiddenDim]
    /// </summary>
    /// <param name="x">Input hidden states.</param>
    /// <param name="positionOffset">
    /// Position of the first token in <paramref name="x"/>.
    /// 0 for full-sequence prefill; kv-cache length for incremental decode.
    /// </param>
    /// <param name="causal">Apply causal (lower-triangular) attention mask.</param>
    public Tensor<float> Forward(Tensor<float> x, int positionOffset = 0, bool causal = true)
    {
        return Forward(x, null, positionOffset, causal);
    }

    public Tensor<float> Forward(Tensor<float> x, KVCache? cache, int positionOffset = 0, bool causal = true)
    {
        ThrowIfDisposed();
        DisposeCache();
        
        _cachedInput = x;
        
        // ── Attention sub-layer ──────────────────────────────────────────
        _cachedNormed1 = _norm1.Forward(x);
        using var normed1 = _cachedNormed1;
        
        _cachedAttnOut = _attention.Forward(normed1, _ops, positionOffset, causal, cache);
        using var attnOut = _cachedAttnOut;

        // Residual: h = x + attn(norm(x))
        _cachedHidden = TensorOps.Add(x, attnOut);
        using var h = _cachedHidden;

        // ── FFN sub-layer ────────────────────────────────────────────────
        _cachedNormed2 = _norm2.Forward(h);
        using var normed2 = _cachedNormed2;
        
        _cachedFfnOut = _ffn.Forward(normed2);
        using var ffnOut = _cachedFfnOut;

        // Residual: out = h + ffn(norm(h))
        return TensorOps.Add(h, ffnOut);
    }
    
    private void DisposeCache()
    {
        _cachedInput?.Dispose();
        _cachedNormed1?.Dispose();
        _cachedAttnOut?.Dispose();
        _cachedHidden?.Dispose();
        _cachedNormed2?.Dispose();
        _cachedFfnOut?.Dispose();
        _cachedInput = null;
        _cachedNormed1 = null;
        _cachedAttnOut = null;
        _cachedHidden = null;
        _cachedNormed2 = null;
        _cachedFfnOut = null;
    }

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var p in _attention.Parameters())
            yield return p;
        foreach (var p in _ffn.Parameters())
            yield return p;
        foreach (var p in _norm1.Parameters())
            yield return p;
        foreach (var p in _norm2.Parameters())
            yield return p;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _attention.Dispose();
        _ffn.Dispose();
        _norm1.Dispose();
        _norm2.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(TransformerBlock));
}