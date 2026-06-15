using SharpMind.Core.Tensors;

namespace SharpMind.Model;

public record TransformerState(int PositionOffset)
{
    public Tensor<int> TokenIds { get; init; } = null!;
    public Tensor<float> Embedded { get; init; } = null!;
    public Tensor<float> Hidden { get; init; } = null!;
    public Tensor<float> Normed { get; init; } = null!;
    public Tensor<float> Logits { get; init; } = null!;
}