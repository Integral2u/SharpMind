using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Training.LoRA;

/// <summary>
/// LoRA applied to attention layers (Q, K, V, O projections).
/// Only the projections listed in <see cref="LoRAConfig.TargetModules"/>
/// (q_proj, k_proj, v_proj, o_proj) are adapted; untargeted projections
/// fall back to the plain frozen matmul.
/// </summary>
public sealed class LoRAAttention : IDisposable
{
    private static readonly string[] ProjectionNames = ["q_proj", "k_proj", "v_proj", "o_proj"];

    private readonly LoRALayer? _loraQ;
    private readonly LoRALayer? _loraK;
    private readonly LoRALayer? _loraV;
    private readonly LoRALayer? _loraO;
    private readonly QuantizationOps _qOps;

    public LoRAAttention(
        int hiddenDim,
        int numHeads,
        int headDim,
        LoRAConfig config)
    {
        int kvDim = numHeads * headDim;
        var targets = new HashSet<string>(config.TargetModules ?? [], StringComparer.OrdinalIgnoreCase);

        _loraQ = targets.Contains("q_proj") ? new LoRALayer(hiddenDim, hiddenDim, config.Rank, config.Scale) : null;
        _loraK = targets.Contains("k_proj") ? new LoRALayer(hiddenDim, kvDim, config.Rank, config.Scale) : null;
        _loraV = targets.Contains("v_proj") ? new LoRALayer(hiddenDim, kvDim, config.Rank, config.Scale) : null;
        _loraO = targets.Contains("o_proj") ? new LoRALayer(hiddenDim, hiddenDim, config.Rank, config.Scale) : null;

        _qOps = QuantizationFactory.Create();
    }

    public Tensor<float> ApplyToQ(Tensor<float> x, Tensor<float> Wq)
        => _loraQ is not null ? _loraQ.Forward(x, Wq) : FrozenForward(x, Wq);

    public Tensor<float> ApplyToK(Tensor<float> x, Tensor<float> Wk)
        => _loraK is not null ? _loraK.Forward(x, Wk) : FrozenForward(x, Wk);

    public Tensor<float> ApplyToV(Tensor<float> x, Tensor<float> Wv)
        => _loraV is not null ? _loraV.Forward(x, Wv) : FrozenForward(x, Wv);

    public Tensor<float> ApplyToO(Tensor<float> x, Tensor<float> Wo)
        => _loraO is not null ? _loraO.Forward(x, Wo) : FrozenForward(x, Wo);

    private unsafe Tensor<float> FrozenForward(Tensor<float> x, Tensor<float> w)
    {
        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);
        using var wBT = w.Transpose();
        var output = new Tensor<float>(x.Shape.Rows, w.Shape.Cols);
        fn(x.DataPtr, (byte*)wBT.DataPtr, output.DataPtr, x.Shape.Rows, x.Shape.Cols, w.Shape.Cols);
        return output;
    }

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
