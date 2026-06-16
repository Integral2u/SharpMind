using SharpMind.Core.Tensors;

namespace SharpMind.Core.Embeddings;

public sealed class AlibiEncoder : PositionalEncoder
{
    public float[] Slopes { get; }

    public AlibiEncoder(int numHeads)
    {
        Slopes = PrecomputeSlopes(numHeads);
    }

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
}
