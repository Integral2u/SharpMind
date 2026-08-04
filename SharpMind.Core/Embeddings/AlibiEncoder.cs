using SharpMind.Core.Tensors;

namespace SharpMind.Core.Embeddings;

/// <summary>
/// ALiBi (Attention with Linear Biases) position encoder.
///
/// Unlike RoPE, ALiBi does NOT modify Q or K tensors — Apply/ApplyBatched are
/// no-ops. Instead, per-head slopes are precomputed here and used later inside
/// the attention kernel (<see cref="Model.Layers.Attention.AttentionKernels"/>)
/// where the bias -slope[h] × (absQPos - j) is added to each Q·K score before
/// softmax. The slope is selected per head from <see cref="Slopes"/> by
/// <c>AttentionLayer.Forward</c> and passed as the <c>alibiSlope</c> parameter.
/// </summary>
public sealed class AlibiEncoder(int numHeads) : PositionalEncoder
{
    /// <summary>Geometric slopes per head: 2^(-8h/nHeads).</summary>
    public float[] Slopes { get; } = PrecomputeSlopes(numHeads);

    private static float[] PrecomputeSlopes(int nHeads)
    {
        var slopes = new float[nHeads];
        for (int h = 1; h <= nHeads; h++)
            slopes[h - 1] = MathF.Pow(2f, -8f * h / nHeads);
        return slopes;
    }

    public override void Apply(Tensor<float> x, int positionOffset = 0)
    {
    }

    public override void ApplyBatched(Tensor<float> x, int positionOffset = 0)
    {
    }

    public override void ApplyBatchedBackward(Tensor<float> dx, int positionOffset = 0)
    {
    }
}
