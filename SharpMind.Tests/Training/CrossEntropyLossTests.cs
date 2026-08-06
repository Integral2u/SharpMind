using SharpMind.Core.Tensors;
using SharpMind.Training.Loss;

namespace SharpMind.Tests.Training;

/// <summary>
/// Verifies label smoothing in <see cref="CrossEntropyLoss"/> (forward value and
/// the combined softmax cross-entropy backward), and that ε = 0 reproduces the
/// existing hard cross-entropy loss/gradient exactly.
/// </summary>
public sealed class CrossEntropyLossTests
{
    private const int IgnoreId = -100;

    private static readonly float[] BaseLogits = [1.0f, 2.0f, 0.5f, -1.0f, 3.0f];

    private static float[] SoftmaxRow(IReadOnlyList<float> row)
    {
        float max = row.Max();
        float sum = row.Sum(v => MathF.Exp(v - max));
        return row.Select(v => MathF.Exp(v - max) / sum).ToArray();
    }

    [Fact]
    public void Forward_MatchesReferenceImplementation()
    {
        using var logits = Tensor<float>.From(BaseLogits, 1, BaseLogits.Length);
        using var labels = Tensor<int>.From([0], 1, 1);
        const float eps = 0.1f;
        var loss = new CrossEntropyLoss(labelSmoothing: eps);

        float actual = loss.Compute(logits, labels);

        // Reference: build p explicitly (no shortcut), then -Σ u[v]·log p[v].
        var p = SoftmaxRow(BaseLogits);
        int V = BaseLogits.Length;
        double expected = 0.0;
        for (int v = 0; v < V; v++)
        {
            double u = v == 0 ? 1 - eps + eps / V : eps / V;
            expected -= u * Math.Log(p[v]);
        }

        Assert.Equal(expected, actual, 4);
    }

    [Fact]
    public void ZeroSmoothing_MatchesPlainLossAndGradient()
    {
        using var logits = Tensor<float>.From(BaseLogits, 1, BaseLogits.Length);
        using var labels = Tensor<int>.From([2], 1, 1);

        var plain = new CrossEntropyLoss();
        var smoothed0 = new CrossEntropyLoss(labelSmoothing: 0f);

        Assert.Equal(plain.Compute(logits, labels), smoothed0.Compute(logits, labels), 6);

        using var dPlain = plain.Backward(logits, labels);
        using var d0 = smoothed0.Backward(logits, labels);
        AssertSpansEqual(dPlain.Data, d0.Data, tolerance: 1e-6f);
    }

    [Fact]
    public void Backward_MatchesFiniteDifference_WithSmoothing()
    {
        float[] logitsData = [1.0f, 2.0f, 0.5f, -1.0f, 3.0f, 0.2f, -0.4f, 1.5f];
        using var logits = Tensor<float>.From(logitsData, 2, 4);
        using var labels = Tensor<int>.From([1, 0], 2, 1);
        const float eps = 0.1f;
        const float h = 1e-3f;
        var loss = new CrossEntropyLoss(labelSmoothing: eps);

        using var d = loss.Backward(logits, labels);
        var engineGrad = d.Data;

        for (int i = 0; i < logitsData.Length; i++)
        {
            float[] plus = (float[])logitsData.Clone(); plus[i] += h;
            float[] minus = (float[])logitsData.Clone(); minus[i] -= h;
            float fd = (ComputeLoss(plus, 2, 4, labels, loss) - ComputeLoss(minus, 2, 4, labels, loss)) / (2 * h);

            float diff = Math.Abs(engineGrad[i] - fd);
            Assert.True(diff <= 1e-3f * (1f + Math.Abs(fd)),
                $"[{i}] backprop={engineGrad[i]:E3} fd={fd:E3} diff={diff:E3}");
        }
    }

    [Fact]
    public void IgnoreId_IsExcluded_WhenSmoothed()
    {
        using var logits = Tensor<float>.From(BaseLogits, 1, BaseLogits.Length);
        using var labels = Tensor<int>.From([IgnoreId], 1, 1);
        var loss = new CrossEntropyLoss(ignoreId: IgnoreId, labelSmoothing: 0.1f);

        Assert.Equal(0f, loss.Compute(logits, labels));

        using var d = loss.Backward(logits, labels);
        Assert.Equal(0f, d.RowSpan(0).ToArray().Sum());
    }

    [Fact]
    public void DefaultNone_IsZero_AndOptionalValuesAccepted()
    {
        Assert.Equal(0f, new CrossEntropyLoss().LabelSmoothing);
        Assert.Equal(0.1f, new CrossEntropyLoss(labelSmoothing: 0.1f).LabelSmoothing);
        Assert.Equal(-100, new CrossEntropyLoss(labelSmoothing: 0.1f).IgnoreId);
    }

    private static float ComputeLoss(
        float[] data, int rows, int cols, Tensor<int> labels, CrossEntropyLoss loss)
    {
        using var logits = Tensor<float>.From(data, rows, cols);
        return loss.Compute(logits, labels);
    }

    /// <summary>Element-wise approximate comparison of two float spans.</summary>
    private static void AssertSpansEqual(ReadOnlySpan<float> a, ReadOnlySpan<float> b, float tolerance)
    {
        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++)
            Assert.True(Math.Abs(a[i] - b[i]) <= tolerance,
                $"element {i}: a={a[i]:E6} b={b[i]:E6}");
    }
}