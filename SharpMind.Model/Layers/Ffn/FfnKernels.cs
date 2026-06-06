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
        TensorOps ops,
        SharpMind.Core.Memory.Workspace? workspace = null)
    {
        using var hidden = w1.Forward(x, ops, workspace);
        using var acted = acts.Activate(hidden, workspace);
        return w2.Forward(acted, ops, workspace);
    }

    /// <summary>
    /// Gated FFN forward: out = (activation(x @ Wgated[0:fD]^T) ⊙ (x @ Wgated[fD:]^T)) @ Wdown^T
    /// WGated is a fused [HiddenDim, 2*FfnDim] layer; gate is first fD columns, up is last fD.
    /// Preserves the batch dimension so WDown.Forward returns [B,S,HiddenDim].
    /// </summary>
    internal static Tensor<float> Gated(
        Tensor<float> x,
        LinearLayer wGated,
        LinearLayer wDown,
        ActivationOps acts,
        TensorOps ops,
        SharpMind.Core.Memory.Workspace? workspace = null)
    {
        using var fused = wGated.Forward(x, ops, workspace);
        int ffnDim = wDown.Weight.Shape[0];
        int[] fusedDims = fused.Shape.Dims.ToArray();
        int total = fused.ElementCount / (2 * ffnDim);
        bool hasBatch = fused.Rank > 2;
        using var gate = workspace != null 
            ? workspace.Rent<float>(hasBatch ? new[] { fusedDims[0], fusedDims[1], ffnDim } : new[] { total, ffnDim })
            : (hasBatch ? new Tensor<float>(fusedDims[0], fusedDims[1], ffnDim) : new Tensor<float>(total, ffnDim));
        using var up = workspace != null 
            ? workspace.Rent<float>(hasBatch ? new[] { fusedDims[0], fusedDims[1], ffnDim } : new[] { total, ffnDim })
            : (hasBatch ? new Tensor<float>(fusedDims[0], fusedDims[1], ffnDim) : new Tensor<float>(total, ffnDim));
        var flat = fused.Reshape(total, 2 * ffnDim);
        for (int i = 0; i < total; i++)
        {
            var row = flat.RowSpan(i);
            row[..ffnDim].CopyTo(gate.RowSpan(i));
            row[ffnDim..].CopyTo(up.RowSpan(i));
        }
        using var gated = acts.GatedActivate(gate, up, workspace);
        return wDown.Forward(gated, ops, workspace);
    }



    /// <summary>
    /// Gated FFN forward with separate gate/up layers (for MoE experts).
    /// </summary>
    internal static Tensor<float> Gated(
        Tensor<float> x,
        LinearLayer wGate,
        LinearLayer wUp,
        LinearLayer wDown,
        ActivationOps acts,
        TensorOps ops,
        SharpMind.Core.Memory.Workspace? workspace = null)
    {
        using var gate = wGate.Forward(x, ops, workspace);
        using var up = wUp.Forward(x, ops, workspace);
        using var gated = acts.GatedActivate(gate, up, workspace);
        return wDown.Forward(gated, ops, workspace);
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
        TensorOps ops,
        SharpMind.Core.Memory.Workspace? workspace = null)
    {
        int batch = x.ElementCount / x.Shape[^1];
        int hidden = x.Shape[^1];
        Tensor<float> result = workspace != null 
            ? workspace.Rent<float>(x.Shape.Dims) 
            : new Tensor<float>(x.Shape);

        // Router logits: [batch, numExperts]
        using var logits = router.Forward(x.Rank > 2 ? x.Reshape(batch, hidden) : x, ops, workspace);
        using var probs = SoftmaxOverExperts(logits, workspace);

        for (int t = 0; t < batch; t++)
        {
            // Get top-k expert indices for this token
            using var tokenLogits = workspace != null 
                ? workspace.Rent<float>(new[] { logits.Shape.Cols }) 
                : Tensor<float>.From(logits.RowSpan(t), logits.Shape.Cols);
            logits.RowSpan(t).CopyTo(tokenLogits.Data);
            int[] topKIdx = TensorOps.ArgTopK(tokenLogits, topK);

            // Accumulate weighted expert outputs
            using var tokenInput = workspace != null 
                ? workspace.Rent<float>(new[] { hidden }) 
                : Tensor<float>.From(x.RowSpan(t), hidden);
            x.RowSpan(t).CopyTo(tokenInput.Data);
            var tokenOut = workspace != null 
                ? workspace.Rent<float>(new[] { hidden }) 
                : Tensor<float>.Zeros(hidden);

            float weightSum = 0f;
            foreach (int expertIdx in topKIdx)
                weightSum += probs.RowSpan(t)[expertIdx];

            foreach (int expertIdx in topKIdx)
            {
                float weight = probs.RowSpan(t)[expertIdx] / weightSum;
                using var expertOut = Gated(tokenInput, wGate[expertIdx],
                                             wUp[expertIdx], wDown[expertIdx], acts, ops, workspace);
                TensorOps.AddInPlace(tokenOut, TensorOps.Scale(expertOut, weight));
            }

            tokenOut.Data.CopyTo(result.RowSpan(t));
            tokenOut.Dispose();
        }

        return result;
    }

    private static Tensor<float> SoftmaxOverExperts(Tensor<float> logits, SharpMind.Core.Memory.Workspace? workspace = null)
    {
        Tensor<float> result = workspace != null 
            ? workspace.Rent<float>(logits.Shape.Dims) 
            : new Tensor<float>(logits.Shape);
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

