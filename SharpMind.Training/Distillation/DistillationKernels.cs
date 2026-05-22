using SharpMind.Core.Tensors;

namespace SharpMind.Training.Distillation;

/// <summary>
/// Knowledge Distillation — transfers knowledge from teacher to student.
/// </summary>
public static class DistillationKernels
{
    /// <summary>
    /// Knowledge Distillation loss combining:
    /// 1. Cross-entropy with hard labels
    /// 2. KL divergence between teacher/student logits (soft labels)
    /// 3. Feature alignment (optional intermediate supervision)
    /// </summary>
    public static float ComputeLoss(
        Tensor<float> studentLogits,
        Tensor<float> teacherLogits,
        Tensor<int> targetLabels,
        float temperature,    // >1 to soften distributions
        float alpha)        // 0=hard only, 1=soft only
    {
        int batchSize = studentLogits.Shape.Rows;
        int vocabSize = studentLogits.Shape.Cols;
        float loss = 0f;

        // Temperature-scaled soft target
        float tScale = temperature;

        for (int b = 0; b < batchSize; b++)
        {
            var studentRow = studentLogits.RowSpan(b);
            var teacherRow = teacherLogits.RowSpan(b);
            int label = targetLabels.RowSpan(b)[0];

            // Hard label cross-entropy
            float hardLoss = CrossEntropy(studentRow, label);

            // Soft target KL divergence
            var studentProbs = Softmax(studentRow, tScale);
            var teacherProbs = Softmax(teacherRow, tScale);
            float softLoss = KLDivergence(studentProbs, teacherProbs);

            loss += (1f - alpha) * hardLoss + alpha * softLoss;
        }

        return loss / batchSize;
    }

    /// <summary>
    /// Intermediate (feature) distillation — aligns intermediate representations.
/// </summary>
    public static float FeatureDistillationLoss(
        Tensor<float> studentFeatures,
        Tensor<float> teacherFeatures,
        Tensor<float> featureAdapter)  // learned adapter
    {
        int batchSize = studentFeatures.Shape.Rows;
        int hiddenDim = studentFeatures.Shape.Cols;

        float loss = 0f;
        for (int b = 0; b < batchSize; b++)
        {
            for (int d = 0; d < hiddenDim; d++)
            {
                float s = studentFeatures.Data[b * hiddenDim + d];
                float t = teacherFeatures.Data[b * hiddenDim + d];
                float a = featureAdapter.Data[d];
                loss += MathF.Pow(s * a - t, 2f);
            }
        }

        return loss / (batchSize * hiddenDim);
    }

    /// <summary>
    /// Weight imitation — student mimics teacher weight gradients.
/// </summary>
    public static float WeightImitationLoss(
        Tensor<float> studentWeights,
        Tensor<float> teacherWeights,
        Tensor<float> studentGrad,
        Tensor<float> teacherGrad,
        float cosineWeight)
    {
        // Cosine similarity on gradients (align update direction)
        float studentNorm = L2Norm(studentGrad);
        float teacherNorm = L2Norm(teacherGrad);

        float cosine = 0f;
        if (studentNorm > 1e-8f && teacherNorm > 1e-8f)
        {
            float dot = 0f;
            for (int i = 0; i < studentGrad.ElementCount; i++)
                dot += studentGrad.Data[i] * teacherGrad.Data[i];
            cosine = dot / (studentNorm * teacherNorm);
        }

        // L2 loss on weight difference
        float l2 = 0f;
        for (int i = 0; i < studentWeights.ElementCount; i++)
            l2 += MathF.Pow(studentWeights.Data[i] - teacherWeights.Data[i], 2f);

        return l2 + cosineWeight * (1f - cosine);
    }

    /// <summary>
    /// Attention transfer — student mimics teacher's attention patterns.
/// </summary>
    public static float AttentionTransferLoss(
        Tensor<float> studentAttn,
        Tensor<float> teacherAttn)
    {
        int batchSize = studentAttn.Shape.Rows;
        int seqLen = studentAttn.Shape.Cols;

        float loss = 0f;
        for (int b = 0; b < batchSize; b++)
        {
            for (int i = 0; i < seqLen; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    float s = studentAttn.Data[b * seqLen * seqLen + i * seqLen + j];
                    float t = teacherAttn.Data[b * seqLen * seqLen + i * seqLen + j];
                    loss += MathF.Pow(s - t, 2f);
                }
            }
        }

        return loss / (batchSize * seqLen * seqLen);
    }

    // Helpers

    private static float CrossEntropy(ReadOnlySpan<float> logits, int label)
    {
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
            sum += MathF.Exp(logits[i] - max);

        return MathF.Log(sum) - (logits[label] - max);
    }

    private static float[] Softmax(ReadOnlySpan<float> logits, float temperature)
    {
        var result = new float[logits.Length];
        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > max) max = logits[i];

        float sum = 0f;
        for (int i = 0; i < logits.Length; i++)
        {
            result[i] = MathF.Exp((logits[i] - max) / temperature);
            sum += result[i];
        }

        for (int i = 0; i < result.Length; i++)
            result[i] /= sum;

        return result;
    }

    private static float KLDivergence(float[] p, float[] q)
    {
        float kl = 0f;
        for (int i = 0; i < p.Length; i++)
        {
            if (p[i] > 1e-10f)
                kl += p[i] * MathF.Log(p[i] / q[i]);
        }
        return kl;
    }

    private static float L2Norm(Tensor<float> t)
    {
        float sum = 0f;
        for (int i = 0; i < t.ElementCount; i++)
            sum += t.Data[i] * t.Data[i];
        return MathF.Sqrt(sum);
    }
}
