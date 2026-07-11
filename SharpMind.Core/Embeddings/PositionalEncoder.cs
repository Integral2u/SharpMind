using SharpMind.Core.Tensors;

namespace SharpMind.Core.Embeddings;

public abstract class PositionalEncoder
{
    public abstract void Apply(Tensor<float> x, int positionOffset = 0);
    public abstract void ApplyBatched(Tensor<float> x, int positionOffset = 0);
}

