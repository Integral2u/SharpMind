using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using SharpMind.Core.Activations;
using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

// FFN kernels — pure static, one unconditional path each

public static class FfnKernels
{
    /// <summary>
    /// A/B switch for the grouped expert path. At batch &gt; 1, tokens routed to
    /// the same expert are gathered so each expert runs one matmul; with the
    /// switch off the layer falls back to the per-token loop (one GEMV per
    /// selected expert). Both compute the same thing — only the weight-read
    /// pattern differs — so this exists to measure the change on one machine.
    /// </summary>
    public static bool GroupExperts { get; set; } = true;

    /// <summary>Thread-local bump allocator for MoE per-token intermediates.</summary>
    private static readonly ThreadLocal<Workspace> t_MoEWorkspace = new(() => new Workspace(1 << 20));
    /// <summary>
    /// Dense FFN forward: out = activation(x @ W1^T + b1) @ W2^T + b2
    /// Applied row-wise — each row is one token's hidden state.
    /// </summary>
    public static Tensor<float> Dense(
        Tensor<float> x,
        LinearLayer w1,
        LinearLayer w2,
        ActivationOps acts,
        SharpMind.Core.Memory.IWorkspace? workspace = null)
    {
        using var hidden = w1.Forward(x, workspace);
        var tAct = DecodeProfiler.Begin();
        using var acted = acts.Activate(hidden, workspace);
        DecodeProfiler.Mark(DecodeStage.Act, tAct);
        return w2.Forward(acted, workspace);
    }

    /// <summary>
    /// Gated FFN forward: out = (activation(x @ Wgated[0:fD]^T) ⊙ (x @ Wgated[fD:]^T)) @ Wdown^T
    /// WGated is a fused [HiddenDim, 2*FfnDim] layer; gate is first fD columns, up is last fD.
    /// Preserves the batch dimension so WDown.Forward returns [B,S,HiddenDim].
    /// </summary>
    public static Tensor<float> Gated(
        Tensor<float> x,
        LinearLayer wGated,
        LinearLayer wDown,
        ActivationOps acts,
        SharpMind.Core.Memory.IWorkspace? workspace = null)
    {
        using var fused = wGated.Forward(x, workspace);
        int ffnDim = wDown.InFeatures;
        int total = fused.ElementCount / (2 * ffnDim);
        bool hasBatch = fused.Rank > 2;
        using var gated = workspace != null 
            ? workspace.Rent<float>(hasBatch ? new[] { fused.Shape[0], fused.Shape[1], ffnDim } : [total, ffnDim])
            : (hasBatch ? new Tensor<float>(fused.Shape[0], fused.Shape[1], ffnDim) : new Tensor<float>(total, ffnDim));
        using var flat = fused.Reshape(total, 2 * ffnDim);

        var tAct = DecodeProfiler.Begin();
        for (int i = 0; i < total; i++)
        {
            var row = flat.RowSpan(i);
            acts.ApplyGate(row[..ffnDim], row[ffnDim..], gated.RowSpan(i));
        }
        DecodeProfiler.Mark(DecodeStage.Act, tAct);

        return wDown.Forward(gated, workspace);
    }



    /// <summary>
    /// Gated FFN forward with separate gate/up layers (for MoE experts).
    /// </summary>
    public static Tensor<float> Gated(
        Tensor<float> x,
        LinearLayer wGate,
        LinearLayer wUp,
        LinearLayer wDown,
        ActivationOps acts,
        SharpMind.Core.Memory.IWorkspace? workspace = null)
    {
        using var gate = wGate.Forward(x, workspace);
        using var up = wUp.Forward(x, workspace);
        var tAct = DecodeProfiler.Begin();
        using var gated = acts.GatedActivate(gate, up, workspace);
        DecodeProfiler.Mark(DecodeStage.Act, tAct);
        var ret = wDown.Forward(gated, workspace);
        return ret;
    }

    /// <summary>
    /// MoE FFN forward: routes each token to top-k experts, computes their
    /// gated FFN outputs, and combines with softmax-normalised router weights.
    /// When <paramref name="residency"/> is non-null the per-token selections are
    /// precomputed as a batch so the hook can load exactly the union of selected
    /// experts before any expert is evaluated; otherwise the non-resident path is
    /// identical to before (routing and evaluation stay interleaved per token).
    /// </summary>
    public static Tensor<float> MoE(
        Tensor<float> x,
        LinearLayer router,
        LinearLayer[] wGate,
        LinearLayer[] wUp,
        LinearLayer[] wDown,
        int topK,
        ActivationOps acts,
        SharpMind.Core.Memory.IWorkspace? workspace = null,
        bool normTopKProb = false,
        IExpertResidencyHost? residency = null,
        Tensor<float>? routerBias = null,
        MoEGatingFunc gating = MoEGatingFunc.Softmax,
        float weightScale = 0f)
    {
        int batch = x.ElementCount / x.Shape[^1];
        int hidden = x.Shape[^1];
        Span<int> dims = stackalloc int[x.Rank];
        x.Shape.CopyDims(dims);
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>(dims)
            : new Tensor<float>(x.Shape);

        // Router logits: [batch, numExperts]
        using var routerInput = x.Rank > 2 ? x.Reshape(batch, hidden) : null;
        using var logits = router.Forward(routerInput ?? x, workspace);
        var tRoute = DecodeProfiler.Begin();
        using var probs = gating == MoEGatingFunc.Sigmoid
            ? SigmoidOverExperts(logits, workspace)
            : SoftmaxOverExperts(logits, workspace);
        // DeepSeek V3 shifts the top-k choice with a per-expert bias but takes the
        // weight from the unbiased probability (llama.cpp build_moe_ffn keeps
        // `probs` untouched and adds the bias to a separate `selection_probs`).
        using var selection = routerBias is null ? null : AddRowBias(probs, routerBias, workspace);
        var selectionProbs = selection ?? probs;
        DecodeProfiler.Mark(DecodeStage.Route, tRoute);

        // Grouped path: several tokens per expert is the only case where a batch
        // can amortise an expert's weights. Each expert then runs one matmul for
        // all rows routed to it instead of one GEMV per token.
        if (batch > 1 && GroupExperts)
        {
            var topIdxAll = new int[batch][];
            var weightSumAll = new float[batch];
            Parallel.For(0, batch, t =>
            {
                using var tokenLogits = Tensor<float>.From(selectionProbs.RowSpan(t), selectionProbs.Shape.Cols);
                int[] topKIdx = ArgTopK(tokenLogits, topK);
                topIdxAll[t] = topKIdx;
                float sum = 0f;
                if (normTopKProb)
                {
                    foreach (int expertIdx in topKIdx) sum += probs.RowSpan(t)[expertIdx];
                    sum = MathF.Max(sum, 6.103515625e-5f);
                }
                weightSumAll[t] = sum;
            });

            if (residency is not null)
            {
                var union = new HashSet<int>();
                foreach (int[] topKIdx in topIdxAll)
                    foreach (int expertIdx in topKIdx) union.Add(expertIdx);
                residency.EnsureExpertsLoaded(union.ToArray());
            }

            return MoEByExpert(x, result, probs, topIdxAll, weightSumAll, wGate, wUp, wDown,
                acts, normTopKProb, weightScale, hidden, batch);
        }

        // Residency path: select all tokens up front so the union can be loaded once, then
        // evaluate. The plain path keeps routing inside ProcessToken (identical to before).
        int[][]? precomputedTopK = null;
        float[]? precomputedSums = null;
        if (residency is not null)
        {
            precomputedTopK = new int[batch][];
            precomputedSums = new float[batch];
            var union = new HashSet<int>();
            for (int t = 0; t < batch; t++)
            {
                using var tokenLogits = Tensor<float>.From(selectionProbs.RowSpan(t), selectionProbs.Shape.Cols);
                int[] topKIdx = ArgTopK(tokenLogits, topK);
                precomputedTopK[t] = topKIdx;
                float weightSum = 0f;
                if (normTopKProb)
                {
                    foreach (int expertIdx in topKIdx) weightSum += probs.RowSpan(t)[expertIdx];
                    weightSum = MathF.Max(weightSum, 6.103515625e-5f);
                }
                precomputedSums[t] = weightSum;
                foreach (int expertIdx in topKIdx) union.Add(expertIdx);
            }
            residency.EnsureExpertsLoaded(union.ToArray());
        }

        // Thread-local bump allocator — one private arena per worker thread,
        // reset at the start of every token, so there is never any contention
        // and no heap allocations for gate/up/gated/down intermediates.
        void ProcessToken(int t)
        {
            var ws = t_MoEWorkspace.Value!;
            ws.Reset();

            // Allocate token input/output from workspace (zero-copy row + zeroed output)
            var tokenInput = ws.Rent<float>([hidden]);
            x.RowSpan(t).CopyTo(tokenInput.Data);

            var tokenOut = ws.Rent<float>([hidden]);
            tokenOut.Data.Clear();

            // Top-k indices and weight sum come from the residency precompute when active.
            int[] topKIdx;
            float weightSum;
            if (precomputedTopK is not null)
            {
                topKIdx = precomputedTopK[t];
                weightSum = precomputedSums![t];
            }
            else
            {
                // Get top-k expert indices
                using var tokenLogits = Tensor<float>.From(selectionProbs.RowSpan(t), selectionProbs.Shape.Cols);
                var tRouteInner = DecodeProfiler.Begin();
                topKIdx = ArgTopK(tokenLogits, topK);

                // Only renormalise when the architecture actually does so. llama.cpp skips
                // this for qwen2moe/olmoe (norm_w = false), leaving the top-k weights as raw
                // softmax probabilities over all experts, which sum to < 1. Dividing by the
                // sum regardless — as this used to — amplified the routed branch and produced
                // gibberish. Clamp to the smallest F16 normal, matching ggml's guard against
                // a zero divisor.
                weightSum = 0f;
                if (normTopKProb)
                {
                    foreach (int expertIdx in topKIdx)
                        weightSum += probs.RowSpan(t)[expertIdx];
                    weightSum = MathF.Max(weightSum, 6.103515625e-5f);
                }
                DecodeProfiler.Mark(DecodeStage.Route, tRouteInner);
            }

            // llama.cpp applies w_scale after renormalising and skips both 0 and 1
            // (DeepSeek V2 writes 1, Moonlight 2.446).
            float scaleW = weightScale is > 0f and not 1f ? weightScale : 1f;

            foreach (int expertIdx in topKIdx)
            {
                float weight = normTopKProb ? probs.RowSpan(t)[expertIdx] / weightSum
                                            : probs.RowSpan(t)[expertIdx];
                weight *= scaleW;
                // Gated uses ws for all intermediates — no heap allocs
                using var expertOut = Gated(tokenInput, wGate[expertIdx],
                                             wUp[expertIdx], wDown[expertIdx], acts, ws);
                // Manual weighted accumulation avoids the allocation in Scale(c)
                var dst = tokenOut.Data;
                var src = expertOut.Data;
                for (int i = 0; i < hidden; i++)
                    dst[i] += src[i] * weight;
            }

            tokenOut.Data.CopyTo(result.RowSpan(t));
        }

        if (batch <= 1)
        {
            for (int t = 0; t < batch; t++)
                ProcessToken(t);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, batch, ProcessToken);
        }

        return result;
    }

    /// <summary>
    /// Evaluates the routed experts in groups: every token routed to expert
    /// <c>e</c> contributes a row to one <c>[m, hidden]</c> matmul through that
    /// expert's weights. At batch 1 each expert sees one row and this is a GEMV,
    /// exactly as before; at batch 8 an expert typically sees several rows and
    /// its weights are read once instead of once per token.
    ///
    /// Rows are gathered into a contiguous input block and expert outputs into a
    /// matching block, then the weighted sum is accumulated per token in top-k
    /// order — the same accumulation the per-token path performs, so switching
    /// paths only changes rounding, not semantics.
    /// </summary>
    private static Tensor<float> MoEByExpert(
        Tensor<float> x,
        Tensor<float> result,
        Tensor<float> probs,
        int[][] topIdxAll,
        float[] weightSumAll,
        LinearLayer[] wGate,
        LinearLayer[] wUp,
        LinearLayer[] wDown,
        ActivationOps acts,
        bool normTopKProb,
        float weightScale,
        int hidden,
        int batch)
    {
        int experts = wGate.Length;
        int assignments = 0;
        foreach (int[] idx in topIdxAll) assignments += idx.Length;

        var counts = new int[experts];
        foreach (int[] idx in topIdxAll)
            foreach (int e in idx) counts[e]++;

        var offsets = new int[experts + 1];
        for (int e = 0; e < experts; e++) offsets[e + 1] = offsets[e] + counts[e];

        var cursor = new int[experts];
        Array.Copy(offsets, cursor, experts);

        var tokenOf = new int[assignments];
        var weightOf = new float[assignments];
        var slotOfToken = new int[batch][];

        float scaleW = weightScale is > 0f and not 1f ? weightScale : 1f;
        for (int t = 0; t < batch; t++)
        {
            var row = probs.RowSpan(t);
            float inv = normTopKProb ? 1f / weightSumAll[t] : 1f;
            int k = topIdxAll[t].Length;
            var slots = new int[k];
            for (int j = 0; j < k; j++)
            {
                int e = topIdxAll[t][j];
                int a = cursor[e]++;
                tokenOf[a] = t;
                weightOf[a] = row[e] * inv * scaleW;
                slots[j] = a;
            }
            slotOfToken[t] = slots;
        }

        using var gathered = new Tensor<float>(assignments, hidden);
        using var expertOutputs = new Tensor<float>(assignments, hidden);
        for (int a = 0; a < assignments; a++)
            x.Data.Slice(tokenOf[a] * hidden, hidden).CopyTo(gathered.Data.Slice(a * hidden, hidden));

        Parallel.For(0, experts, e =>
        {
            int m = counts[e];
            if (m == 0) return;
            int start = offsets[e];

            using var input = new Tensor<float>(m, hidden);
            gathered.Data.Slice(start * hidden, m * hidden).CopyTo(input.Data);

            using var produced = Gated(input, wGate[e], wUp[e], wDown[e], acts);
            produced.Data.CopyTo(expertOutputs.Data.Slice(start * hidden, m * hidden));
        });

        var outputData = expertOutputs.Data;
        var resultData = result.Data;
        resultData.Clear();
        for (int t = 0; t < batch; t++)
        {
            var dst = resultData.Slice(t * hidden, hidden);
            foreach (int a in slotOfToken[t])
            {
                float w = weightOf[a];
                var src = outputData.Slice(a * hidden, hidden);
                for (int i = 0; i < hidden; i++)
                    dst[i] += src[i] * w;
            }
        }

        return result;
    }

    /// <summary>
    /// Per-expert sigmoid (DeepSeek V3's <c>expert_gating_func = 2</c>), with no
    /// cross-expert normalisation: the selected top-k are renormalised later only
    /// when the file asks for it.
    /// </summary>
    public static Tensor<float> SigmoidOverExperts(Tensor<float> logits, SharpMind.Core.Memory.IWorkspace? workspace = null)
    {
        Span<int> dims = stackalloc int[logits.Rank];
        logits.Shape.CopyDims(dims);
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>(dims)
            : new Tensor<float>(logits.Shape);
        for (int i = 0; i < logits.Shape.Rows; i++)
        {
            var src = logits.RowSpan(i);
            var dst = result.RowSpan(i);
            for (int j = 0; j < src.Length; j++) dst[j] = 1f / (1f + MathF.Exp(-src[j]));
        }
        return result;
    }

    /// <summary>
    /// Adds a per-expert vector to every row, producing the tensor the top-k
    /// selection runs on when the file carries a router selection bias.
    /// </summary>
    public static Tensor<float> AddRowBias(
        Tensor<float> probs, Tensor<float> bias, SharpMind.Core.Memory.IWorkspace? workspace = null)
    {
        if (bias.ElementCount != probs.Shape.Cols)
            throw new ArgumentException(
                $"Router bias has {bias.ElementCount} entries but the router produced {probs.Shape.Cols} experts.",
                nameof(bias));
        Span<int> dims = stackalloc int[probs.Rank];
        probs.Shape.CopyDims(dims);
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>(dims)
            : new Tensor<float>(probs.Shape);
        var biasData = bias.Data;
        for (int i = 0; i < probs.Shape.Rows; i++)
        {
            var src = probs.RowSpan(i);
            var dst = result.RowSpan(i);
            for (int j = 0; j < src.Length; j++) dst[j] = src[j] + biasData[j];
        }
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<float> SoftmaxOverExperts(Tensor<float> logits, SharpMind.Core.Memory.IWorkspace? workspace = null)
    {
        Span<int> dims = stackalloc int[logits.Rank];
        logits.Shape.CopyDims(dims);
        Tensor<float> result = workspace != null
            ? workspace.Rent<float>(dims)
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
    public static int[] ArgTopK(Tensor<float> a, int k)
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

    public static int[] ArgTopKIntroselectArray(ReadOnlySpan<float> data, int k)
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
