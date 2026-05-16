namespace SharpMind.Model.Layers;

public sealed class RmsNormLayer(int dim, float eps = 1e-5f) : NormLayer(dim, hasBias: false, eps)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
        => NormKernels.RMSNormRowScalar(src, weight, dst, rmsInv);

    protected override float ComputeScalarParam(ReadOnlySpan<float> row)
    {
        // Guard against overflow: when |v| > sqrt(float.MaxValue) ≈ 1.84e19,
        // v² overflows to +Inf, making the norm output all zeros.
        // Normalize by maxAbs to keep all intermediate squares in range.
        float maxAbs = 0f;
        foreach (float v in row)
        {
            float a = Math.Abs(v);
            if (a > maxAbs) maxAbs = a;
        }

        if (maxAbs < 1e-20f)
            return 1f / MathF.Sqrt(Eps);

        float invMax = 1f / maxAbs;
        float ss = 0f;
        foreach (float v in row)
        {
            float vn = v * invMax;
            ss += vn * vn;
        }
        // mean(x²) = ss / N * maxAbs²
        // rms = maxAbs * sqrt(ss / N + eps / maxAbs²)
        float rms = maxAbs * MathF.Sqrt(ss / row.Length + Eps / (maxAbs * maxAbs));
        return 1f / rms;
    }

    protected override void ComputeBackward(float[] input, float[] output, ReadOnlySpan<float> dOutput, Span<float> dInput, int N, float eps, float rmsInv)
    {
        float rms = 1f / rmsInv;
        for (int i = 0; i < N; i++) dInput[i] = dOutput[i] * rms;
    }
}
