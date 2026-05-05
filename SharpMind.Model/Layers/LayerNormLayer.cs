namespace SharpMind.Model.Layers;

public sealed class LayerNormLayer(int dim, float eps = 1e-5f) : NormLayer(dim, hasBias: true, eps)
{
    public override void ApplyRow(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float eps)
    {
        float mean = 0f;
        foreach (float v in src) mean += v;
        mean /= src.Length;
        float variance = 0f;
        foreach (float v in src) { float d = v - mean; variance += d * d; }
        variance /= src.Length;
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
        variance /= N;
        float invStd = 1f / MathF.Sqrt(variance + eps);
        for (int i = 0; i < N; i++) dInput[i] = dOutput[i];
        float sumDY = 0f;
        for (int i = 0; i < N; i++) sumDY += dOutput[i];
        sumDY /= N;
        for (int i = 0; i < N; i++) dInput[i] -= sumDY;
    }
}