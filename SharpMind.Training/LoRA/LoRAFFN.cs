using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Core.Ops;

namespace SharpMind.Training.LoRA;

/// <summary>
/// LoRA applied to FFN layer.
/// </summary>
public sealed class LoRAFFN(int hiddenDim, int ffnDim, LoRAConfig config) : IDisposable
{
    private readonly LoRALayer _gate = new(hiddenDim, ffnDim, config.Rank, config.Scale);
    private readonly LoRALayer _up = new(hiddenDim, ffnDim, config.Rank, config.Scale);
    private readonly LoRALayer _down = new(ffnDim, hiddenDim, config.Rank, config.Scale);

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
