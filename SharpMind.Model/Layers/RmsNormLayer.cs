using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

public sealed class RmsNormLayer(int dim, float eps = 1e-5f, Tensor<float>? weight = null, Tensor<float>? bias = null) : NormLayer(dim, false, eps, weight, bias)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
        => NormKernels.RMSNormRowScalar(src, weight, dst, rmsInv);

    protected override float ComputeScalarParam(ReadOnlySpan<float> row)
    {
        // Single-pass online RMS with overflow guard.
        // When |v| > sqrt(float.MaxValue) ≈ 1.84e19, v² overflows to +Inf.
        // Normalize by running maxAbs to keep all intermediate squares in range.
        float maxAbs = 0f;
        float ss = 0f;
        int n = row.Length;

        foreach (float v in row)
        {
            float a = Math.Abs(v);
            if (a > maxAbs)
            {
                float ratio = maxAbs / a;
                ss = ss * ratio * ratio + 1f;
                maxAbs = a;
            }
            else if (a > 1e-20f)
            {
                float vn = v / maxAbs;
                ss += vn * vn;
            }
        }

        if (maxAbs < 1e-20f)
            return 1f / MathF.Sqrt(Eps);

        float rms = maxAbs * MathF.Sqrt(ss / n + Eps / (maxAbs * maxAbs));
        return 1f / rms;
    }

    protected override void ComputeBackward(float[] input, float[] output, ReadOnlySpan<float> dOutput, Span<float> dInput, int N, float eps, float rmsInv)
    {
        float rms = 1f / rmsInv;
        for (int i = 0; i < N; i++) dInput[i] = dOutput[i] * rms;
    }
}
