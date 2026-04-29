namespace SharpMind.Model.Layers;

/// <summary>
/// LayerNorm concrete subclass — passes eps as the scalar parameter;
/// the kernel computes mean/variance internally.
/// </summary>
public sealed class LayerNormLayer(int dim, float eps = 1e-5f) : NormLayer(dim, hasBias: true, eps)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight,
                                  Span<float> dst, float eps)
        => throw new InvalidOperationException(
            "ApplyRow must be overridden by the JigSaw-assembled type.");

    protected override float ComputeScalarParam(ReadOnlySpan<float> row) => Eps;
}