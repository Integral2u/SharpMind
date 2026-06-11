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
    public Tensor<float> Forward(Tensor<float> x, int positionOffset = 0, bool causal = true, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        return Forward(x, null, positionOffset, causal, workspace);
    }

    public Tensor<float> Forward(Tensor<float> x, IKVCache? cache, int positionOffset = 0, bool causal = true, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();
        
        // ── Attention sub-layer ──────────────────────────────────────────
        var normed1 = _norm1.Forward(x, workspace);
        var attnOut = _attention.Forward(normed1, _ops, positionOffset, causal, cache, workspace);
        normed1.Dispose();
        
        // Residual: h = x + attn(norm(x)) — reuse x in-place
        TensorOps.AddInPlace(x, attnOut);
        attnOut.Dispose();
        
        // ── FFN sub-layer ────────────────────────────────────────────────
        var normed2 = _norm2.Forward(x, workspace);
        var ffnOut = _ffn.Forward(normed2, workspace);
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
        // Attention — supports both old (attn_q) and modern (q_proj/self_attn.q) naming
        if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("self_attn.q", StringComparison.OrdinalIgnoreCase))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("self_attn.k", StringComparison.OrdinalIgnoreCase))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("self_attn.v", StringComparison.OrdinalIgnoreCase))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_o", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("o_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("out_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("self_attn.o", StringComparison.OrdinalIgnoreCase))
        {
            _attention.LoadWeights(name, data);
            return true;
        }

        // FFN — supports both old (ffn_gate) and modern (gate_proj/mlp.gate) naming
        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("gate_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("mlp.gate", StringComparison.OrdinalIgnoreCase))
        {
            _ffn.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) || name.Contains("up_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("mlp.up", StringComparison.OrdinalIgnoreCase))
        {
            _ffn.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) || name.Contains("down_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("mlp.down", StringComparison.OrdinalIgnoreCase))
        {
            _ffn.LoadWeights(name, data);
            return true;
        }

        // Norms — supports both old and modern naming
        if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase))
        {
            _norm1.LoadWeight(data);
            return true;
        }
        if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase))
        {
            _norm2.LoadWeight(data);
            return true;
        }

        return false;
    }

    public bool SetRawWeight(string name, byte[] rawData, Format.GgufDtype dtype)
    {
        if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_o", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("q_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("o_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("out_proj", StringComparison.OrdinalIgnoreCase))
            return _attention.SetRawWeight(name, rawData, dtype);
        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("gate_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("up_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("down_proj", StringComparison.OrdinalIgnoreCase))
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