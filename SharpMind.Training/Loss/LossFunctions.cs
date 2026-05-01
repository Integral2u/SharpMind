using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Training.Autograd;

namespace SharpMind.Training.Loss;

/// <summary>
/// Causal language model cross-entropy loss.
///
/// Computes mean negative log-likelihood over non-ignored positions:
///   L = -1/N * sum_t log(softmax(logits[t])[label[t]])
///
/// Positions where label == <see cref="IgnoreId"/> (-100 by default) are
/// excluded from both the sum and the count N. This matches the convention
/// used by <c>PackingBatcher</c> which writes -100 into padding and the
/// last position of each packed sequence.
///
/// The backward pass is always computed as the combined softmax + cross-entropy
/// gradient — never as separate softmax backward then cross-entropy backward.
/// The softmax Jacobian is [VocabSize, VocabSize] per token — intractable at
/// vocab sizes of 32k–128k. The combined gradient collapses to:
///   dL/dLogits[t, v] = (softmax(logits[t])[v] - 1{v == label[t]}) / N
/// which is what <see cref="Backward"/> returns.
/// </summary>
public sealed class CrossEntropyLoss : ILoss<int>
{
    public int IgnoreId { get; }

    public CrossEntropyLoss(int ignoreId = -100) => IgnoreId = ignoreId;

    /// <summary>
    /// Computes the scalar loss.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public float Compute(Tensor<float> logits, Tensor<int> labels)
    {
        int T = logits.Shape.Rows;
        int V = logits.Shape.Cols;

        if (labels.ElementCount != T)
            throw new ArgumentException(
                $"Label count {labels.ElementCount} must match logit rows {T}.");

        double totalLoss = 0.0;
        int realCount = 0;

        for (int t = 0; t < T; t++)
        {
            if (labels[t] == IgnoreId) continue;
            realCount++;

            ReadOnlySpan<float> row = logits.RowSpan(t);
            float max = row[0];
            for (int v = 1; v < V; v++) if (row[v] > max) max = row[v];

            float sumExp = 0f;
            for (int v = 0; v < V; v++) sumExp += MathF.Exp(row[v] - max);

            float logProb = (row[labels[t]] - max) - MathF.Log(sumExp);
            totalLoss -= logProb;
        }

        return realCount > 0 ? (float)(totalLoss / realCount) : 0f;
    }

    /// <summary>
    /// Computes dL/dLogits as the combined softmax + cross-entropy gradient.
    /// Returns [T, VocabSize] — pass directly to the LM head backward.
    ///
    /// Computing softmax backward and cross-entropy backward separately is
    /// intractable — the softmax Jacobian is [V, V] per token. The combined
    /// form collapses to a simple elementwise operation and is always used.
    /// </summary>
    public Tensor<float> Backward(Tensor<float> logits, Tensor<int> labels)
        => Gradients.CrossEntropySoftmax(logits, labels, IgnoreId);
}

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
