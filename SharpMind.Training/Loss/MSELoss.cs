using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Training.Loss;

/// <summary>
/// Mean squared error loss: L = mean((predictions - targets)²).
/// Used for regression tasks and distillation targets.
/// </summary>
public sealed class MSELoss : ILoss<float>
{
    public float Compute(Tensor<float> predictions, Tensor<float> targets)
    {
        if (predictions.ElementCount != targets.ElementCount)
            throw new ArgumentException(
                $"Prediction count {predictions.ElementCount} must match target count {targets.ElementCount}.");

        double sum = 0.0;
        var p = predictions.Data;
        var t = targets.Data;
        for (int i = 0; i < p.Length; i++) { double d = p[i] - t[i]; sum += d * d; }
        return (float)(sum / p.Length);
    }

    /// <summary>
    /// MSE backward: dL/dpredictions = 2 * (predictions - targets) / N
    /// </summary>
    public Tensor<float> Backward(Tensor<float> predictions, Tensor<float> targets)
    {
        var dOut = new Tensor<float>(predictions.Shape);
        var p = predictions.Data;
        var t = targets.Data;
        var dst = dOut.Data;
        float inv = 2f / p.Length;
        for (int i = 0; i < p.Length; i++) dst[i] = inv * (p[i] - t[i]);
        return dOut;
    }
}
