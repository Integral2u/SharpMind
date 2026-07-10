using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Training.LoRA;

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

    public Tensor<float> ApplyToQ(Tensor<float> x, Tensor<float> Wq)
        => _loraQ!.Forward(x, Wq);

    public Tensor<float> ApplyToK(Tensor<float> x, Tensor<float> Wk)
        => _loraK!.Forward(x, Wk);

    public Tensor<float> ApplyToV(Tensor<float> x, Tensor<float> Wv)
        => _loraV!.Forward(x, Wv);

    public Tensor<float> ApplyToO(Tensor<float> x, Tensor<float> Wo)
        => _loraO!.Forward(x, Wo);

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
