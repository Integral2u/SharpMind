using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers.Ffn;

/// <summary>Training state saved during FFN forward pass.</summary>
public class FfnLayerState
{
    public Tensor<float> Input { get; init; } = null!;
    public Tensor<float> Output { get; init; } = null!;
    public FfnKind Kind { get; init; }
}

