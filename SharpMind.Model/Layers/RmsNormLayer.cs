namespace SharpMind.Model.Layers;

public sealed class RmsNormLayer(int dim, float eps = 1e-5f) : NormLayer(dim, hasBias: false, eps)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
        => NormKernels.RMSNormRowScalar(src, weight, dst, rmsInv);

    protected override float ComputeScalarParam(ReadOnlySpan<float> row)
    {
        float ss = 0f;
        foreach (float v in row) ss += v * v;
        return 1f / MathF.Sqrt(ss / row.Length + Eps);
    }

    protected override void ComputeBackward(float[] input, float[] output, ReadOnlySpan<float> dOutput, Span<float> dInput, int N, float eps, float rmsInv)
    {
        float rms = 1f / rmsInv;
        for (int i = 0; i < N; i++) dInput[i] = dOutput[i] * rms;
    }
}
