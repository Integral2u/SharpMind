using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Training.Autograd;

namespace SharpMind.Training.Loss;

/// <summary>
/// Causal language model cross-entropy loss with optional label smoothing.
///
/// Computes mean negative log-likelihood over non-ignored positions:
///   L = -1/N * sum_t log(softmax(logits[t])[label[t]])
///
/// With label smoothing epsilon ε > 0 the one-hot target is replaced by
///   u[label] = 1 - ε + ε/V,   u[v≠label] = ε/V
/// giving   L = -1/N * sum_t [ (1-ε)·log p[label] + (ε/V)·Σ_v log p[v] ].
/// The Σ_v log p[v] term is computed without a per-token log() via the identity
///   Σ_v log p[v] = (Σ_v logits[v]) - V·max - V·log(sumExp).
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
///   dL/dLogits[t, v] = (softmax(logits[t])[v] - u[v]) / N
/// which is what <see cref="Backward"/> returns.
/// </summary>
public sealed class CrossEntropyLoss(int ignoreId = -100, float labelSmoothing = 0.0f) : ILoss<int>
{
    public int IgnoreId { get; init; } = ignoreId;

    /// <summary>Label smoothing epsilon. 0 = plain one-hot cross-entropy.</summary>
    public float LabelSmoothing { get; init; } = labelSmoothing;


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
        ArgumentOutOfRangeException.ThrowIfLessThan(LabelSmoothing, 0f, nameof(LabelSmoothing));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(LabelSmoothing, 1f, nameof(LabelSmoothing));

        double totalLoss = 0.0;
        int realCount = 0;

        if (LabelSmoothing <= 0f)
        {
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
        }
        else
        {
            float smooth = LabelSmoothing;
            float epsOverV = smooth / V;
            float oneMinusEps = 1f - smooth;

            for (int t = 0; t < T; t++)
            {
                if (labels[t] == IgnoreId) continue;
                realCount++;

                ReadOnlySpan<float> row = logits.RowSpan(t);
                float max = row[0];
                float sumRow = row[0];
                for (int v = 1; v < V; v++) { if (row[v] > max) max = row[v]; sumRow += row[v]; }

                float sumExp = 0f;
                for (int v = 0; v < V; v++) sumExp += MathF.Exp(row[v] - max);

                float logProbLabel = (row[labels[t]] - max) - MathF.Log(sumExp);
                // Σ_v log p[v] = (Σ_v row[v]) - V·max - V·log(sumExp)
                float sumLogProb = sumRow - V * max - V * MathF.Log(sumExp);

                totalLoss -= oneMinusEps * logProbLabel + epsOverV * sumLogProb;
            }
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
        => Gradients.CrossEntropySoftmax(logits, labels, IgnoreId, LabelSmoothing);
}
