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

    public NormLayer Norm1 => _norm1;
    public NormLayer Norm2 => _norm2;
    public AttentionLayer Attention => _attention;
    public FfnLayer Ffn => _ffn;
    private readonly TensorOps _ops;
    private readonly int _layerIdx;
    private bool _disposed;

    private Tensor<float>? _cachedInput;
    private Tensor<float>? _cachedNormed1;
    private Tensor<float>? _cachedAttnOut;
    private Tensor<float>? _cachedHidden;
    private Tensor<float>? _cachedNormed2;
    private Tensor<float>? _cachedFfnOut;

    public TransformerBlock(
        int layerIdx,
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

        _layerIdx = layerIdx;
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
        
        // ── Attention sub-layer ──────────────────────────────────────────
        var normed1 = _norm1.Forward(x);
        var attnOut = _attention.Forward(normed1, _ops, positionOffset, causal, cache);
        normed1.Dispose();
        
        // Residual: h = x + attn(norm(x)) — reuse x in-place
        TensorOps.AddInPlace(x, attnOut);
        attnOut.Dispose();
        
        // ── FFN sub-layer ────────────────────────────────────────────────
        var normed2 = _norm2.Forward(x);
        var ffnOut = _ffn.Forward(normed2);
        normed2.Dispose();
        
        // Residual: out = h + ffn(norm(h)) — reuse x in-place
        TensorOps.AddInPlace(x, ffnOut);
        ffnOut.Dispose();
        
        return x;
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

    public bool LoadWeight(string name, ReadOnlySpan<float> data)
    {
        var lower = name.ToLower();
        
        if (lower.Contains("attn_q"))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
        if (lower.Contains("attn_k"))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
        if (lower.Contains("attn_v"))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
        if (lower.Contains("attn_output") || lower.Contains("attn_o"))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
        if (lower.Contains("ffn_gate"))
        {
            _ffn.LoadWeights(name, data);
            return true;
        }
        if (lower.Contains("ffn_up"))
        {
            _ffn.LoadWeights(name, data);
            return true;
        }
        if (lower.Contains("ffn_down"))
        {
            _ffn.LoadWeights(name, data);
            return true;
        }
        if (lower.Contains("attn_norm"))
        {
            _norm1.LoadWeight(data);
            return true;
        }
        if (lower.Contains("ffn_norm"))
        {
            _norm2.LoadWeight(data);
            return true;
        }
        
        return false;
    }

    public bool SetRawWeight(string name, byte[] rawData, Format.GgufDtype dtype)
    {
        var lower = name.ToLower();
        if (lower.Contains("attn_q") || lower.Contains("attn_k") || lower.Contains("attn_v") ||
            lower.Contains("attn_output") || lower.Contains("attn_o"))
            return _attention.SetRawWeight(name, rawData, dtype);
        if (lower.Contains("ffn_gate") || lower.Contains("ffn_up") || lower.Contains("ffn_down"))
            return _ffn.SetRawWeight(name, rawData, dtype);
        return false;
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