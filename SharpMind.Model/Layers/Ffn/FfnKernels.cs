using System.Threading.Tasks;
using SharpMind.Core.Activations;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers.Ffn;

// FFN kernels — pure static, one unconditional path each

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
        SharpMind.Core.Memory.Workspace? workspace = null)
    {
        using var hidden = w1.Forward(x, workspace);
        using var acted = acts.Activate(hidden, workspace);
        return w2.Forward(acted, workspace);
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
        SharpMind.Core.Memory.Workspace? workspace = null)
    {
        using var fused = wGated.Forward(x, workspace);
        int ffnDim = wDown.InFeatures;
        int[] fusedDims = fused.Shape.Dims.ToArray();
        int total = fused.ElementCount / (2 * ffnDim);
        bool hasBatch = fused.Rank > 2;
        using var gated = workspace != null 
            ? workspace.Rent<float>(hasBatch ? new[] { fusedDims[0], fusedDims[1], ffnDim } : [total, ffnDim])
            : (hasBatch ? new Tensor<float>(fusedDims[0], fusedDims[1], ffnDim) : new Tensor<float>(total, ffnDim));
        var flat = fused.Reshape(total, 2 * ffnDim);

        for (int i = 0; i < total; i++)
        {
            var row = flat.RowSpan(i);
            acts.ApplyGate(row[..ffnDim], row[ffnDim..], gated.RowSpan(i));
        }

        return wDown.Forward(gated, workspace);
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
        SharpMind.Core.Memory.Workspace? workspace = null)
    {
        using var gate = wGate.Forward(x, workspace);
        using var up = wUp.Forward(x, workspace);
        using var gated = acts.GatedActivate(gate, up, workspace);
        return wDown.Forward(gated, workspace);
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
        SharpMind.Core.Memory.Workspace? workspace = null)
    {
        int batch = x.ElementCount / x.Shape[^1];
        int hidden = x.Shape[^1];
        Tensor<float> result = workspace != null 
            ? workspace.Rent<float>(x.Shape.Dims) 
            : new Tensor<float>(x.Shape);

        // Router logits: [batch, numExperts]
        using var logits = router.Forward(x.Rank > 2 ? x.Reshape(batch, hidden) : x, workspace);
        using var probs = SoftmaxOverExperts(logits, workspace);

        // Parallel token processing — no workspace sharing to avoid races
        System.Threading.Tasks.Parallel.For(0, batch, t =>
        {
            // Get top-k expert indices for this token (fresh tensor, no workspace)
            using var tokenLogits = Tensor<float>.From(logits.RowSpan(t), logits.Shape.Cols);
            int[] topKIdx = ArgTopK(tokenLogits, topK);

            // Accumulate weighted expert outputs
            using var tokenInput = Tensor<float>.From(x.RowSpan(t), hidden);
            using var tokenOut = Tensor<float>.Zeros(hidden);

            float weightSum = 0f;
            foreach (int expertIdx in topKIdx)
                weightSum += probs.RowSpan(t)[expertIdx];

            foreach (int expertIdx in topKIdx)
            {
                float weight = probs.RowSpan(t)[expertIdx] / weightSum;
                using var expertOut = Gated(tokenInput, wGate[expertIdx],
                                             wUp[expertIdx], wDown[expertIdx], acts);
                tokenOut.AddInPlace(expertOut.Scale(weight));
            }

            tokenOut.Data.CopyTo(result.RowSpan(t));
        });

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

    /// <summary>Returns the flat indices of the top <paramref name="k"/> elements (sorted descending).</summary>
    private static int[] ArgTopK(Tensor<float> a, int k)
    {
        if (k <= 0 || k > a.ElementCount)
            throw new ArgumentOutOfRangeException(nameof(k), $"k={k} must be in [1, {a.ElementCount}].");

        int n = a.ElementCount;
        ReadOnlySpan<float> data = a.Data;

        if (k >= n)
        {
            var indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i;
            float[] dataArr = a.Data.ToArray();
            Array.Sort(indices, (x, y) => dataArr[y].CompareTo(dataArr[x]));
            var result = new int[k];
            Array.Copy(indices, result, k);
            return result;
        }

        if (k <= 64)
        {
            var pq = new PriorityQueue<int, float>();
            for (int i = 0; i < n; i++)
            {
                float val = data[i];
                if (pq.Count < k)
                {
                    pq.Enqueue(i, val);
                }
                else if (pq.TryPeek(out _, out float minPriority) && val > minPriority)
                {
                    pq.Dequeue();
                    pq.Enqueue(i, val);
                }
            }
            var result = new int[k];
            for (int i = k - 1; i >= 0; i--) result[i] = pq.Dequeue();
            return result;
        }

        return ArgTopKIntroselectArray(data, k);
    }

    private static int[] ArgTopKIntroselectArray(ReadOnlySpan<float> data, int k)
    {
        int n = data.Length;
        var indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        int left = 0, right = n - 1;
        int target = k - 1;

        while (left < right)
        {
            int pivot = PartitionArray(data, indices, left, right);
            if (pivot == target) break;
            if (pivot > target) right = pivot - 1;
            else left = pivot + 1;
        }

        var result = new int[k];
        for (int i = 0; i < k; i++) result[i] = indices[i];
        float[] dataArr = data.ToArray();
        Array.Sort(result, (x, y) => dataArr[y].CompareTo(dataArr[x]));
        return result;
    }

    private static int PartitionArray(ReadOnlySpan<float> data, int[] indices, int left, int right)
    {
        float pivot = data[indices[left]];
        int i = left;
        for (int j = left + 1; j <= right; j++)
        {
            if (data[indices[j]] > pivot)
            {
                i++;
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }
        (indices[i], indices[left]) = (indices[left], indices[i]);
        return i;
    }
}
