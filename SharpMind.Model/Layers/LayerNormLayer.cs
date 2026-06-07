using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

public sealed class LayerNormLayer(int dim, float eps = 1e-5f, Tensor<float>? weight = null, Tensor<float>? bias = null) : NormLayer(dim, true, eps, weight, bias)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float eps)
    {
        float mean = 0f;
        foreach (float v in src) mean += v;
        mean /= src.Length;
        // Guard against overflow when squaring large values.
        float maxAbs = 0f;
        foreach (float v in src) { float d = v - mean; float a = Math.Abs(d); if (a > maxAbs) maxAbs = a; }
        float variance;
        if (maxAbs < 1e-20f)
        {
            variance = 0f;
        }
        else
        {
            float invMax = 1f / maxAbs;
            float sum = 0f;
            foreach (float v in src) { float d = (v - mean) * invMax; sum += d * d; }
            variance = sum / src.Length * maxAbs * maxAbs;
        }
        float invStd = 1f / MathF.Sqrt(variance + eps);
        for (int i = 0; i < dst.Length; i++)
            dst[i] = (src[i] - mean) * invStd * weight[i] + Bias!.Data[i];
    }

    protected override float ComputeScalarParam(ReadOnlySpan<float> row)
    {
        float mean = 0f;
        foreach (float v in row) mean += v;
        return mean / row.Length;
    }

    protected override void ComputeBackward(float[] input, float[] output, ReadOnlySpan<float> dOutput, Span<float> dInput, int N, float eps, float storedMean)
    {
        float mean = storedMean;
        float variance = 0f;
        foreach (float v in input) { float d = v - mean; variance += d * d; }
        for (int i = 0; i < N; i++) dInput[i] = dOutput[i];
        float sumDY = 0f;
        for (int i = 0; i < N; i++) sumDY += dOutput[i];
        sumDY /= N;
        for (int i = 0; i < N; i++) dInput[i] -= sumDY;
    }
}