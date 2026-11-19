using SharpMind.Core.Tensors;

namespace SharpMind.Core.Embeddings;

public abstract class PositionalEncoder
{
    public abstract void Apply(Tensor<float> x, int positionOffset = 0);
    public abstract void ApplyBatched(Tensor<float> x, int positionOffset = 0);

    /// <summary>
    /// Applies a batched tensor [Batch, SeqLen, NumHeads, HeadDim] whose rows
    /// start at different positions. A batch that mixes sequences at different
    /// cache lengths needs this: one <paramref name="rowPositions"/> entry per
    /// row replaces the single position offset. Encoders that ignore positions
    /// (NoPE, ALiBi) inherit the no-op below.
    /// </summary>
    public virtual void ApplyBatched(Tensor<float> x, ReadOnlySpan<int> rowPositions)
    {
        if (x.Rank != 4)
            throw new ArgumentException(
                $"ApplyBatched expects rank-4 [Batch, SeqLen, NumHeads, HeadDim], got rank {x.Rank}.");

        int batch = x.Shape[0];
        if (rowPositions.Length < batch)
            throw new ArgumentException(
                $"Expected {batch} row positions, got {rowPositions.Length}.", nameof(rowPositions));

        for (int b = 0; b < batch; b++)
        {
            using var slice = x.Slice(b);
            Apply(slice, rowPositions[b]);
        }
    }

    /// <summary>
    /// Backward of <see cref="ApplyBatched"/> (in-place inverse rotation of the
    /// gradient). Takes the gradient w.r.t. the post-rotated tensor and mutates
    /// it in place into the gradient w.r.t. the pre-rotation tensor. Encoders
    /// that do not modify Q/K during forward (e.g. NoPE, ALiBi) are no-ops.
    /// </summary>
    public abstract void ApplyBatchedBackward(Tensor<float> dx, int positionOffset = 0);
}

