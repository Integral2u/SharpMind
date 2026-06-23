using SharpMind.Core.Memory;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.Model.Layers;

public abstract class TransformerBlock : IDisposable
{
    protected readonly AttentionLayer _attention;
    protected readonly FfnLayer _ffn;
    protected readonly NormLayer _norm1;
    protected readonly NormLayer _norm2;
    protected readonly TensorOps _ops;
    protected readonly int _layerIdx;
    private bool _disposed;

    public NormLayer Norm1 => _norm1;
    public NormLayer Norm2 => _norm2;
    public AttentionLayer Attention => _attention;
    public FfnLayer Ffn => _ffn;

    protected TransformerBlock(int layerIdx, AttentionLayer attention, FfnLayer ffn, NormLayer norm1, NormLayer norm2, TensorOps ops)
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

    public abstract Tensor<float> Forward(Tensor<float> x, IKVCache? cache, int positionOffset = 0, bool causal = true, Workspace? workspace = null);

    public Tensor<float> Forward(Tensor<float> x, int positionOffset = 0, bool causal = true, Workspace? workspace = null)
        => Forward(x, null, positionOffset, causal, workspace);

    public virtual void SetActivationHook(IActivationHook? hook) { }

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
        // Q/K norm checks must precede the broader attn_q/attn_k checks
        if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
        if (name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase))
        {
            _attention.LoadWeights(name, data);
            return true;
        }
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
        // Q/K norm (always loaded as float) — skip quantized path
        if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase) ||
            name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase))
            return false;
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

    protected void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(TransformerBlock));
}
