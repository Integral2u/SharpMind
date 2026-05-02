namespace SharpMind.Model.Layers;

/// <summary>
/// LayerNorm concrete subclass — passes eps as the scalar parameter;
/// the kernel from <see cref="NormKernels"/> computes mean/variance internally.
/// </summary>
public sealed class LayerNormLayer(int dim, float eps = 1e-5f) : NormLayer(dim, hasBias: true, eps)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight,
                                  Span<float> dst, float eps)
    {
        // Compute mean and variance
        float mean = 0f;
        foreach (float v in src) mean += v;
        mean /= src.Length;

        float variance = 0f;
        foreach (float v in src) { float d = v - mean; variance += d * d; }
        variance /= src.Length;

        float invStd = 1f / MathF.Sqrt(variance + eps);
        
        // Apply: (x - mean) * invStd * weight + bias
        for (int i = 0; i < dst.Length; i++)
            dst[i] = (src[i] - mean) * invStd * weight[i] + Bias!.Data[i];
    }

    protected override float ComputeScalarParam(ReadOnlySpan<float> row) => Eps;
}