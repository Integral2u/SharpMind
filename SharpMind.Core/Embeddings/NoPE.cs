using SharpMind.Core.Tensors;

namespace SharpMind.Core.Embeddings;

public sealed class NoPE : PositionalEncoder
{
    public override void Apply(Tensor<float> x, int positionOffset = 0)
    {
    }

    public override void ApplyBatched(Tensor<float> x, int positionOffset = 0)
    {
    }
}
