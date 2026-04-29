namespace SharpMind.Model.Layers;

/// <summary>
/// RMSNorm concrete subclass — overrides <see cref="NormLayer.ComputeScalarParam"/>
/// to compute rmsInv. The JigSaw-assembled <see cref="NormLayer.ApplyRow"/> handles
/// the kernel dispatch.
/// </summary>
public sealed class RmsNormLayer(int dim, float eps = 1e-5f) : NormLayer(dim, hasBias: false, eps)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight,
                                  Span<float> dst, float rmsInv)
        => throw new InvalidOperationException(
            "ApplyRow must be overridden by the JigSaw-assembled type.");

    protected override float ComputeScalarParam(ReadOnlySpan<float> row)
    {
        float ss = 0f;
        foreach (float v in row) ss += v * v;
        return 1f / MathF.Sqrt(ss / row.Length + Eps);
    }
}
