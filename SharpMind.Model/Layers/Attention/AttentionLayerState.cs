using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers.Attention;

public class AttentionLayerState
{
    public Tensor<float> Input { get; init; } = null!;
    public Tensor<float> Output { get; init; } = null!;
}