using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Core.Ops;
using SharpMind.Model;
using SharpMind.Model.Layers;

namespace SharpMind.Training.LoRA;

/// <summary>
/// LoRA (Low-Rank Adaptation) — parameter-efficient fine-tuning.
/// Adds small rank decomposition matrices to frozen pretrained weights.
/// W_fast = W_frozen + (A @ B.T) * scale
/// 
/// This allows fine-tuning with ~1% of original parameters while
/// retaining most of model quality.
/// </summary>
public sealed class LoRALayer : IDisposable
{
    private readonly LinearLayer _a;  // [rank, in]
    private readonly LinearLayer _b;  // [out, rank]
    private readonly float _scale;
    private bool _disposed;

    public LoRALayer(int inFeatures, int outFeatures, int rank, float scale = 1f)
    {
        if (rank > inFeatures) rank = inFeatures;
        if (rank > outFeatures) rank = outFeatures;

        Rank = rank;
        _scale = scale;

        _a = new LinearLayer(rank, inFeatures, bias: false);
        _b = new LinearLayer(outFeatures, rank, bias: false);

        // Initialize with small random values (Gauss-Ortho init style)
        var rng = Random.Shared;
        for (int i = 0; i < _a.Weight.ElementCount; i++)
            _a.Weight.Data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;
        for (int i = 0; i < _b.Weight.ElementCount; i++)
            _b.Weight.Data[i] = (float)(rng.NextDouble() * 2 - 1) * 0.01f;
    }

    public int Rank { get; }
    public int InFeatures => _a.InFeatures;
    public int OutFeatures => _b.OutFeatures;

    /// <summary>
    /// Applies LoRA adaptation to frozen weights.
    /// Output = frozen(x) + scale * (B @ A) @ x
    /// </summary>
    public Tensor<float> Forward(Tensor<float> input, Tensor<float> frozenWeights, TensorOps ops)
    {
        ThrowIfDisposed();

        // frozen_output = input @ W_frozen^T
        var frozen = ops.MatMul(input, frozenWeights);

        // lora_adjustment = input @ A^T @ B^T
        // A: [rank, in], B: [out, rank]
        // First: input @ A^T -> [batch, rank]
        using var afterA = ops.MatMul(input, TensorOps.Transpose(_a.Weight));
        // Then: afterA @ B^T -> [batch, out]
        var loraAdjust = ops.MatMul(afterA, TensorOps.Transpose(_b.Weight));

        // Combine: frozen + scale * lora
        var result = TensorOps.Add(frozen, TensorOps.Scale(loraAdjust, _scale));
        
        frozen.Dispose();
        loraAdjust.Dispose();

        return result;
    }

    /// <summary>
    /// Applies LoRA to embed → hidden transform.
    /// </summary>
    public Tensor<float> ForwardEmbedding(TensorOps ops)
    {
        // W_fast = W_embed + (B @ A) * scale
        var lora = ops.MatMul(_a.Weight, _b.Weight);
        return TensorOps.Scale(lora, _scale);
    }

    public IEnumerable<Parameter> Parameters()
    {
        yield return new Parameter("lora.a", _a.Weight);
        yield return new Parameter("lora.b", _b.Weight);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _a.Dispose();
        _b.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(LoRALayer));
}

/// <summary>
/// LoRA config for whole model transformation.
/// </summary>
public class LoRAConfig
{
    public int Rank { get; set; } = 8;
    public float Alpha { get; set; } = 16f;  // often rank * 2
    public float Dropout { get; set; } = 0.0f;
    public string[] TargetModules { get; set; } = ["q_proj", "v_proj", "k_proj", "o_proj"];

    public float Scale => Alpha / Rank;
}

/// <summary>
/// LoRA applied to attention layers (Q, K, V, O projections).
/// </summary>
public sealed class LoRAAttention : IDisposable
{
    private readonly LoRALayer? _loraQ;
    private readonly LoRALayer? _loraK;
    private readonly LoRALayer? _loraV;
    private readonly LoRALayer? _loraO;

    public LoRAAttention(
        int hiddenDim,
        int numHeads,
        int headDim,
        LoRAConfig config)
    {
        int kvDim = numHeads * headDim;

        // Apply LoRA to all attention projections
        _loraQ = new LoRALayer(hiddenDim, hiddenDim, config.Rank, config.Scale);
        _loraK = new LoRALayer(hiddenDim, kvDim, config.Rank, config.Scale);
        _loraV = new LoRALayer(hiddenDim, kvDim, config.Rank, config.Scale);
        _loraO = new LoRALayer(hiddenDim, hiddenDim, config.Rank, config.Scale);
    }

    public Tensor<float> ApplyToQ(Tensor<float> x, Tensor<float> Wq, TensorOps ops)
        => _loraQ!.Forward(x, Wq, ops);

    public Tensor<float> ApplyToK(Tensor<float> x, Tensor<float> Wk, TensorOps ops)
        => _loraK!.Forward(x, Wk, ops);

    public Tensor<float> ApplyToV(Tensor<float> x, Tensor<float> Wv, TensorOps ops)
        => _loraV!.Forward(x, Wv, ops);

    public Tensor<float> ApplyToO(Tensor<float> x, Tensor<float> Wo, TensorOps ops)
        => _loraO!.Forward(x, Wo, ops);

    public IEnumerable<Parameter> Parameters()
    {
        if (_loraQ is not null) foreach (var p in _loraQ.Parameters()) yield return p;
        if (_loraK is not null) foreach (var p in _loraK.Parameters()) yield return p;
        if (_loraV is not null) foreach (var p in _loraV.Parameters()) yield return p;
        if (_loraO is not null) foreach (var p in _loraO.Parameters()) yield return p;
    }

    public void Dispose()
    {
        _loraQ?.Dispose();
        _loraK?.Dispose();
        _loraV?.Dispose();
        _loraO?.Dispose();
    }
}

/// <summary>
/// LoRA applied to FFN layer.
/// </summary>
public sealed class LoRAFFN : IDisposable
{
    private readonly LoRALayer _gate;
    private readonly LoRALayer _up;
    private readonly LoRALayer _down;

    public LoRAFFN(int hiddenDim, int ffnDim, LoRAConfig config)
    {
        _gate = new LoRALayer(hiddenDim, ffnDim, config.Rank, config.Scale);
        _up = new LoRALayer(hiddenDim, ffnDim, config.Rank, config.Scale);
        _down = new LoRALayer(ffnDim, hiddenDim, config.Rank, config.Scale);
    }

    public Tensor<float> ApplyGate(Tensor<float> x, Tensor<float> Wgate, TensorOps ops)
        => _gate.Forward(x, Wgate, ops);

    public Tensor<float> ApplyUp(Tensor<float> x, Tensor<float> Wup, TensorOps ops)
        => _up.Forward(x, Wup, ops);

    public Tensor<float> ApplyDown(Tensor<float> x, Tensor<float> Wdown, TensorOps ops)
        => _down.Forward(x, Wdown, ops);

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var p in _gate.Parameters()) yield return p;
        foreach (var p in _up.Parameters()) yield return p;
        foreach (var p in _down.Parameters()) yield return p;
    }

    public void Dispose()
    {
        _gate.Dispose();
        _up.Dispose();
        _down.Dispose();
    }
}

/// <summary>
/// Full model with LoRA adaptation.
/// </summary>
public class LoRAModel : IDisposable
{
    private readonly Transformer _baseModel;
    private readonly LoRAAttention _attention;
    private readonly LoRAFFN? _ffn;
    private readonly LoRAConfig _config;
    private bool _disposed;

    public LoRAModel(Transformer baseModel, LoRAConfig config)
    {
        _baseModel = baseModel;
        _config = config;

        _attention = new LoRAAttention(
            baseModel.Config.HiddenDim,
            baseModel.Config.NumHeads,
            baseModel.Config.HeadDim,
            config);

        if (baseModel.Config.FfnDim > 0)
        {
            _ffn = new LoRAFFN(
                baseModel.Config.HiddenDim,
                baseModel.Config.FfnDim,
                config);
        }
    }

    public IEnumerable<Parameter> LoRAParameters()
    {
        foreach (var p in _attention.Parameters())
            yield return p;
        if (_ffn is not null)
            foreach (var p in _ffn.Parameters())
                yield return p;
    }

    /// <summary>
    /// Number of trainable parameters vs original.
    /// </summary>
    public double TrainableRatio()
    {
        long baseParams = _baseModel.ParameterCount;
        long loraParams = LoRAParameters().Count() * _config.Rank;  // rough estimate
        return (double)loraParams / baseParams;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _attention.Dispose();
        _ffn?.Dispose();
    }
}