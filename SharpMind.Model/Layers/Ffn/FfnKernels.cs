using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers.Ffn;

// ─────────────────────────────────────────────────────────────────────────────
// FFN kernels — pure static, one unconditional path each
// ─────────────────────────────────────────────────────────────────────────────

internal static class FfnKernels
{
    /// <summary>
    /// Dense FFN forward: out = activation(x @ W1^T + b1) @ W2^T + b2
    /// Applied row-wise — each row is one token's hidden state.
    /// </summary>
    internal static Tensor<float> Dense(
        Tensor<float> x,
        LinearLayer w1,
        LinearLayer w2,
        ActivationOps acts,
        TensorOps ops)
    {
        using var hidden = w1.Forward(x, ops);
        using var acted = acts.Activate(hidden);
        return w2.Forward(acted, ops);
    }

    /// <summary>
    /// Gated FFN forward: out = (activation(x @ Wgate^T) ⊙ (x @ Wup^T)) @ Wdown^T
    /// SwiGLU / GeGLU pattern used in LLaMA, Mistral, PaLM.
    /// </summary>
    internal static Tensor<float> Gated(
        Tensor<float> x,
        LinearLayer wGate,
        LinearLayer wUp,
        LinearLayer wDown,
        ActivationOps acts,
        TensorOps ops)
    {
        using var gate = wGate.Forward(x, ops);
        using var up = wUp.Forward(x, ops);
        using var gated = acts.GatedActivate(gate, up);
        return wDown.Forward(gated, ops);
    }

    /// <summary>
    /// MoE FFN forward: routes each token to top-k experts, computes their
    /// gated FFN outputs, and combines with softmax-normalised router weights.
    /// </summary>
    internal static Tensor<float> MoE(
        Tensor<float> x,
        LinearLayer router,
        LinearLayer[] wGate,
        LinearLayer[] wUp,
        LinearLayer[] wDown,
        int topK,
        ActivationOps acts,
        TensorOps ops)
    {
        int batch = x.ElementCount / x.Shape[^1];
        int hidden = x.Shape[^1];
        var result = new Tensor<float>(x.Shape);

        // Router logits: [batch, numExperts]
        using var logits = router.Forward(x.Rank > 2 ? x.Reshape(batch, hidden) : x, ops);
        using var probs = SoftmaxOverExperts(logits);

        for (int t = 0; t < batch; t++)
        {
            // Get top-k expert indices for this token
            using var tokenLogits = Tensor<float>.From(logits.RowSpan(t), logits.Shape.Cols);
            int[] topKIdx = TensorOps.ArgTopK(tokenLogits, topK);

            // Accumulate weighted expert outputs
            using var tokenInput = Tensor<float>.From(x.RowSpan(t), hidden);
            var tokenOut = Tensor<float>.Zeros(hidden);

            float weightSum = 0f;
            foreach (int expertIdx in topKIdx)
                weightSum += probs.RowSpan(t)[expertIdx];

            foreach (int expertIdx in topKIdx)
            {
                float weight = probs.RowSpan(t)[expertIdx] / weightSum;
                using var expertOut = Gated(tokenInput, wGate[expertIdx],
                                            wUp[expertIdx], wDown[expertIdx], acts, ops);
                TensorOps.AddInPlace(tokenOut, TensorOps.Scale(expertOut, weight));
            }

            tokenOut.Data.CopyTo(result.RowSpan(t));
            tokenOut.Dispose();
        }

        return result;
    }

    private static Tensor<float> SoftmaxOverExperts(Tensor<float> logits)
    {
        var result = new Tensor<float>(logits.Shape);
        for (int i = 0; i < logits.Shape.Rows; i++)
        {
            var src = logits.RowSpan(i);
            var dst = result.RowSpan(i);
            float max = src[0];
            foreach (float v in src) if (v > max) max = v;
            float sum = 0f;
            for (int j = 0; j < src.Length; j++) { dst[j] = MathF.Exp(src[j] - max); sum += dst[j]; }
            float inv = 1f / sum;
            for (int j = 0; j < dst.Length; j++) dst[j] *= inv;
        }
        return result;
    }
}

