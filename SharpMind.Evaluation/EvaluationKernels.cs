using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Evaluation;

/// <summary>
/// Model evaluation metrics.
/// </summary>
public static class EvaluationKernels
{
    /// <summary>
    /// Perplexity — measures how well the model predicts the next token.
    /// Lower is better. exp(loss)
    /// </summary>
    public static float Perplexity(Tensor<float> logits, Tensor<int> targetIds)
    {
        int vocabSize = logits.Shape.Cols;
        int seqLen = logits.Shape.Cols;
        int batch = logits.Shape.Rows;

        float totalLoss = 0f;
        int tokenCount = 0;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int tokenId = targetIds.RowSpan(b * seqLen + s)[0];
                if (tokenId >= vocabSize) continue;

                var probs = Softmax(logits.RowSpan(b * seqLen + s));
                float prob = probs[tokenId];

                if (prob > 1e-10f)
                    totalLoss -= MathF.Log(prob);

                tokenCount++;
            }
        }

        if (tokenCount == 0) return float.PositiveInfinity;

        float avgLoss = totalLoss / tokenCount;
        return MathF.Exp(avgLoss);
    }

    /// <summary>
    /// Token-level accuracy — does the model predict the exact next token?
    /// </summary>
    public static (int Correct, int Total) Accuracy(
        Tensor<float> logits,
        Tensor<int> targetIds,
        int topK = 1)
    {
        int vocabSize = logits.Shape.Cols;
        int seqLen = logits.Shape.Cols;
        int batch = logits.Shape.Rows;

        int correct = 0;
        int total = 0;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int target = targetIds.RowSpan(b * seqLen + s)[0];
                if (target >= vocabSize) continue;

                var row = logits.RowSpan(b * seqLen + s);
                var topIndices = ArgMaxK(row, topK);

                foreach (var idx in topIndices)
                {
                    if (idx == target)
                    {
                        correct++;
                        break;
                    }
                }

                total++;
            }
        }

        return (correct, total);
    }

    /// <summary>
    /// Exact match — does the entire sequence match?
    /// </summary>
    public static float ExactMatch(Tensor<float> logits, Tensor<int> targetIds)
    {
        int seqLen = logits.Shape.Cols;
        int batch = logits.Shape.Rows;

        int matches = 0;

        for (int b = 0; b < batch; b++)
        {
            bool match = true;
            for (int s = 0; s < seqLen; s++)
            {
                int target = targetIds.RowSpan(b * seqLen + s)[0];
                int predicted = ArgMax(logits.RowSpan(b * seqLen + s));

                if (predicted != target)
                {
                    match = false;
                    break;
                }
            }

            if (match) matches++;
        }

        return (float)matches / batch;
    }

    /// <summary>
    /// BLEU score for sequence generation.
    /// </summary>
    public static float BleuScore(
        int[] reference,
        int[] hypothesis,
        int nGram = 4)
    {
        if (reference.Length == 0 || hypothesis.Length == 0)
            return 0f;

        var refNgrams = new Dictionary<string, int>();
        var hypNgrams = new Dictionary<string, int>();

        int matches = 0;
        int hypCount = 0;

        for (int n = 1; n <= nGram; n++)
        {
            for (int i = 0; i <= reference.Length - n; i++)
            {
                var ng = string.Join("_", reference.Skip(i).Take(n));
                refNgrams[ng] = refNgrams.GetValueOrDefault(ng) + 1;
            }

            for (int i = 0; i <= hypothesis.Length - n; i++)
            {
                var ng = string.Join("_", hypothesis.Skip(i).Take(n));
                hypNgrams[ng] = hypNgrams.GetValueOrDefault(ng) + 1;
            }

            foreach (var kvp in hypNgrams)
            {
                int m = Math.Min(kvp.Value, refNgrams.GetValueOrDefault(kvp.Key));
                matches += m;
            }

            hypCount += hypothesis.Length - n + 1;
        }

        if (hypCount == 0) return 0f;

        float precision = matches / hypCount;
        float brevity = reference.Length > 0
            ? MathF.Min(1f, hypothesis.Length / (float)reference.Length)
            : 0f;

        return precision * MathF.Pow(brevity, 0.25f);
    }

    /// <summary>
    /// F1 score for classification.
    /// </summary>
    public static (float Precision, float Recall, float F1) ClassificationMetrics(
        Tensor<float> logits,
        Tensor<int> targetIds)
    {
        int vocabSize = logits.Shape.Cols;
        int batch = logits.Shape.Rows;

        int truePositives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;

        for (int b = 0; b < batch; b++)
        {
            int target = targetIds.RowSpan(b)[0];
            int predicted = ArgMax(logits.RowSpan(b));

            if (predicted == target)
            {
                if (predicted < vocabSize)
                    truePositives++;
            }
            else
            {
                if (predicted < vocabSize)
                    falsePositives++;
                if (target < vocabSize)
                    falseNegatives++;
            }
        }

        float precision = truePositives + falsePositives > 0
            ? truePositives / (float)(truePositives + falsePositives)
            : 0f;

        float recall = truePositives + falseNegatives > 0
            ? truePositives / (float)(truePositives + falseNegatives)
            : 0f;

        float f1 = precision + recall > 0
            ? 2 * precision * recall / (precision + recall)
            : 0f;

        return (precision, recall, f1);
    }

    // Helpers

    private static float[] Softmax(ReadOnlySpan<float> logits)
    {
        var result = new float[logits.Length];
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = MathF.Exp(logits[i] - max);
            sum += result[i];
        }

        for (int i = 0; i < result.Length; i++)
            result[i] /= sum;

        return result;
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int maxIdx = 0;
        float maxVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > maxVal)
            {
                maxVal = values[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    private static int[] ArgMaxK(ReadOnlySpan<float> values, int k)
    {
        var indices = new int[values.Length];
        for (int i = 0; i < indices.Length; i++) indices[i] = i;

        Array.Sort(indices, (a, b) => values[b].CompareTo(values[a]));

        return indices.Take(k).ToArray();
    }
}

/// <summary>
/// Evaluation metrics for full generation tasks.
/// </summary>
public class EvaluationResult
{
    public float Perplexity { get; set; }
    public float TokenAccuracy { get; set; }
    public float ExactMatch { get; set; }
    public (float Precision, float Recall, float F1) Classification { get; set; }
    public float bleuScore { get; set; }
}