using SharpMind.Core.Memory;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.DeltaNet;
using SharpMind.Model.Layers.Ffn;
using SharpMind.Model.Layers.ShortConv;

namespace SharpMind.Model.Layers;

public abstract class TransformerBlock : IDisposable
{
    // Exactly one of these is non-null: attention blocks carry _attention,
    // LFM2 short-conv blocks carry _shortConv, Qwen3.5 linear layers carry _deltaNet.
    protected readonly AttentionLayer? _attention;
    protected readonly ShortConvLayer? _shortConv;
    protected readonly DeltaNetLayer? _deltaNet;
    protected readonly FfnLayer _ffn;
    protected readonly NormLayer _norm1;
    protected readonly NormLayer _norm2;
    protected NormLayer? _postAttnNorm;
    protected NormLayer? _postFfnNorm;
    protected readonly int _layerIdx;
    // Gemma-4 per-layer-embedding tail and layer output scale (null elsewhere).
    protected readonly PerLayerEmbedding? _perLayer;
    protected readonly Tensor<float>? _outScale;
    // BERT-style post-norm residual placement: no pre-norm; norm1/norm2 are
    // applied after each residual add instead of to the sublayer input.
    protected readonly bool _postNormResidual;
    private bool _disposed;

    public NormLayer Norm1 => _norm1;
    public NormLayer Norm2 => _norm2;
    public AttentionLayer? Attention => _attention;
    public ShortConvLayer? ShortConv => _shortConv;
    public DeltaNetLayer? DeltaNet => _deltaNet;
    public FfnLayer Ffn => _ffn;
    public NormLayer? PostAttnNorm => _postAttnNorm;
    public NormLayer? PostFfnNorm => _postFfnNorm;
    public bool PostNormResidual => _postNormResidual;

    protected TransformerBlock(int layerIdx, AttentionLayer? attention, FfnLayer ffn, NormLayer norm1, NormLayer norm2,
        NormLayer? postAttnNorm = null, NormLayer? postFfnNorm = null, ShortConvLayer? shortConv = null, bool postNormResidual = false,
        PerLayerEmbedding? perLayer = null, Tensor<float>? outScale = null, DeltaNetLayer? deltaNet = null)
    {
        ArgumentNullException.ThrowIfNull(ffn);
        ArgumentNullException.ThrowIfNull(norm1);
        ArgumentNullException.ThrowIfNull(norm2);
        if (attention is null && shortConv is null && deltaNet is null)
            throw new ArgumentException("A transformer block requires either an attention, short-conv, or delta-net branch.");

        _layerIdx = layerIdx;
        _attention = attention;
        _shortConv = shortConv;
        _deltaNet = deltaNet;
        _ffn = ffn;
        _norm1 = norm1;
        _norm2 = norm2;
        _postAttnNorm = postAttnNorm;
        _postFfnNorm = postFfnNorm;
        _postNormResidual = postNormResidual;
        _perLayer = perLayer;
        _outScale = outScale;
    }

    /// <summary>
    /// Runs the block's first (attention or short-conv) branch on a normed input.
    /// Both branches add their output to the residual later in <c>Forward</c>.
    /// </summary>
    protected Tensor<float> FirstBranch(Tensor<float> normed, IKVCache? cache, int positionOffset, bool causal, IWorkspace? workspace, int windowSize)
        => _shortConv is not null
            ? _shortConv.Forward(normed, cache, workspace)
            : _deltaNet is not null
                ? _deltaNet.Forward(normed, cache, workspace)
                : _attention!.Forward(normed, positionOffset, causal, cache, workspace, windowSize);

    public abstract Tensor<float> Forward(Tensor<float> x, IKVCache? cache, int positionOffset, bool causal, IWorkspace? workspace, int windowSize, Tensor<float>? perLayerInputs);

    public Tensor<float> Forward(Tensor<float> x, IKVCache? cache, int positionOffset = 0, bool causal = true, IWorkspace? workspace = null, int windowSize = 0)
        => Forward(x, cache, positionOffset, causal, workspace, windowSize, null);

    public Tensor<float> Forward(Tensor<float> x, int positionOffset = 0, bool causal = true, IWorkspace? workspace = null, int windowSize = 0)
        => Forward(x, null, positionOffset, causal, workspace, windowSize, null);

    /// <summary>
    /// Applies the gemma-4 per-layer-embedding tail (when this block has one) followed
    /// by the layer output scale. Called at the end of every block forward.
    /// </summary>
    protected Tensor<float> ApplyPerLayerTail(Tensor<float> x, Tensor<float>? perLayerInputs, IWorkspace? workspace)
    {
        if (_perLayer is not null && perLayerInputs is not null)
            _perLayer.ApplyInPlace(x, perLayerInputs, workspace);
        if (_outScale is not null)
            x.ScaleInPlace(_outScale.Data[0]);
        return x;
    }

    public virtual void SetActivationHook(IActivationHook? hook) { }

    public IEnumerable<Parameter> Parameters()
    {
        if (_attention != null)
            foreach (var p in _attention.Parameters())
                yield return p;
        if (_shortConv != null)
            foreach (var p in _shortConv.Parameters())
                yield return p;
        if (_deltaNet != null)
            foreach (var p in _deltaNet.Parameters())
                yield return p;
        foreach (var p in _ffn.Parameters())
            yield return p;
        foreach (var p in _norm1.Parameters())
            yield return p;
        foreach (var p in _norm2.Parameters())
            yield return p;
        if (_postAttnNorm != null)
            foreach (var p in _postAttnNorm.Parameters())
                yield return p;
        if (_postFfnNorm != null)
            foreach (var p in _postFfnNorm.Parameters())
                yield return p;
    }

    public bool LoadWeight(string name, ReadOnlySpan<float> data)
    {
        // Delta-net tensors must be checked before the generic attn_q/attn_output
        // matches below ("attn_qkv" contains "attn_q").
        if (_deltaNet is not null &&
            (name.Contains("ssm_", StringComparison.OrdinalIgnoreCase) ||
             name.Contains("attn_qkv", StringComparison.OrdinalIgnoreCase) ||
             name.Contains("attn_gate", StringComparison.OrdinalIgnoreCase)))
        {
            _deltaNet.LoadWeights(name, data);
            return true;
        }
        // Short-conv tensors must be checked before the generic attn_out_proj
        // / out_proj matches below (llama names shortconv.out_proj).
        if (name.Contains("shortconv", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("sc_in_proj", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("sc_out_proj", StringComparison.OrdinalIgnoreCase))
        {
            _shortConv?.LoadWeights(name, data);
            return true;
        }
        // Q/K norm checks must precede the broader attn_q/attn_k checks
        if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase))
        {
            _attention?.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase))
        {
            _attention?.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("self_attn.q", StringComparison.OrdinalIgnoreCase))
        {
            _attention?.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("self_attn.k", StringComparison.OrdinalIgnoreCase))
        {
            _attention?.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("self_attn.v", StringComparison.OrdinalIgnoreCase))
        {
            _attention?.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_output_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) _norm1.LoadBias(data);
            else _norm1.LoadWeight(data);
            return true;
        }
        if (name.Contains("layer_output_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) _norm2.LoadBias(data);
            else _norm2.LoadWeight(data);
            return true;
        }
        if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_o", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("o_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("self_attn.o", StringComparison.OrdinalIgnoreCase))
        {
            _attention?.LoadWeights(name, data);
            return true;
        }
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
        if (name.Contains("post_attention_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) return false;
            // Gemma-3 applies this norm to the attention output. Qwen3.5 has no
            // post-attention norm; its post_attention_norm is the pre-FFN norm.
            if (_postAttnNorm is not null) _postAttnNorm.LoadWeight(data);
            else _norm2.LoadWeight(data);
            return true;
        }
        if (name.Contains("post_ffw_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) return false;
            _postFfnNorm?.LoadWeight(data);
            return true;
        }
        if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) _norm1.LoadBias(data);
            else _norm1.LoadWeight(data);
            return true;
        }
        if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) _norm2.LoadBias(data);
            else _norm2.LoadWeight(data);
            return true;
        }

        return false;
    }

    public bool SetRawWeight(string name, byte[] rawData, QuantDType dtype)
    {
        if (_deltaNet is not null &&
            (name.Contains("ssm_", StringComparison.OrdinalIgnoreCase) ||
             name.Contains("attn_qkv", StringComparison.OrdinalIgnoreCase) ||
             name.Contains("attn_gate", StringComparison.OrdinalIgnoreCase)))
            return _deltaNet.SetRawWeight(name, rawData, dtype);
        // Short-conv checks must precede the generic out_proj matches below.
        if (name.Contains("shortconv", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("sc_in_proj", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("sc_out_proj", StringComparison.OrdinalIgnoreCase))
            return _shortConv?.SetRawWeight(name, rawData, dtype) ?? false;
        // Q/K norm (always loaded as float) — skip quantized path
        if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase))
            return false;
        if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_o", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("q_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("o_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("out_proj", StringComparison.OrdinalIgnoreCase))
            return _attention?.SetRawWeight(name, rawData, dtype) ?? false;
        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("gate_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("up_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("down_proj", StringComparison.OrdinalIgnoreCase))
            return _ffn.SetRawWeight(name, rawData, dtype);
        return false;
    }

    public void SetWeights(TransformerWeights.BlockWeights weights)
    {
        _attention?.SetWeights(weights);
        _shortConv?.SetWeights(weights);
        _deltaNet?.SetWeights(weights);
        _ffn.SetWeights(weights);
        if (weights.Norm1W != null)
            _norm1.LoadWeight(weights.Norm1W.Data);
        if (weights.Norm1B != null && _norm1.NormBias != null)
            _norm1.LoadBias(weights.Norm1B.Data);
        if (weights.Norm2W != null)
            _norm2.LoadWeight(weights.Norm2W.Data);
        if (weights.Norm2B != null && _norm2.NormBias != null)
            _norm2.LoadBias(weights.Norm2B.Data);
    }

    public void FreeFloatWeights()
    {
        _attention?.FreeFloatWeights();
        _shortConv?.FreeFloatWeights();
        _deltaNet?.FreeFloatWeights();
        _ffn.FreeFloatWeights();
    }

    /// <summary>Disables the Q8_0 wide repack on this block's projections. Streaming calls
    /// this once; the repack would otherwise rebuild on every layer reload.</summary>
    public void SuppressWide()
    {
        _attention?.SuppressWide();
        _shortConv?.SuppressWide();
        _deltaNet?.SuppressWide();
        _ffn.SuppressWide();
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        _disposed = true;
        if (disposing)
        {
            _attention?.Dispose();
            _shortConv?.Dispose();
            _deltaNet?.Dispose();
            _ffn.Dispose();
            _norm1.Dispose();
            _norm2.Dispose();
            _postAttnNorm?.Dispose();
            _postFfnNorm?.Dispose();
            _perLayer?.Dispose();
        }
    }

    protected void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(TransformerBlock));
}
