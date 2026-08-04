using SharpMind.Core.Tensors;

namespace SharpMind.Core.Embeddings;

public abstract class PositionalEncoder
{
    public abstract void Apply(Tensor<float> x, int positionOffset = 0);
    public abstract void ApplyBatched(Tensor<float> x, int positionOffset = 0);

    /// <summary>
    /// Backward of <see cref="ApplyBatched"/> (in-place inverse rotation of the
    /// gradient). Takes the gradient w.r.t. the post-rotated tensor and mutates
    /// it in place into the gradient w.r.t. the pre-rotation tensor. Encoders
    /// that do not modify Q/K during forward (e.g. NoPE, ALiBi) are no-ops.
    /// </summary>
    public abstract void ApplyBatchedBackward(Tensor<float> dx, int positionOffset = 0);
}

