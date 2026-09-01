using ILGPU;
using ILGPU.Algorithms;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// Causal attention that never materialises the probabilities, computing the same values as
/// <see cref="AttentionKernels"/> from an online (streaming) softmax.
///
/// <see cref="AttentionKernels"/> keeps <c>P [B·H·S, S]</c> per block for the whole step so the
/// backward can read it — at 24 blocks that is the term which grows as S² and dominates the
/// arena at long sequences. Here the forward keeps only per query row the softmax statistics
/// <c>m = max_j s_ij</c> and <c>l = Σ_j exp(s_ij − m)</c> — a factor of S smaller — and the
/// backward recomputes <c>p_ij = exp(s_ij − m_i)/l_i</c> from them. That is the flash-attention
/// trade: score arithmetic twice, an S² array never.
///
/// The three per-row scalars share one <c>stats [B·H·S, 3]</c> tensor — m, l, and the backward's
/// row constant D — rather than three views, because ILGPU's kernel loaders cap how many
/// arguments a launch may take and the backward kernels are already near it.
///
/// The accumulator lives in the destination row rather than in registers: ILGPU kernels cannot
/// size a local array from a runtime headDim, and every thread here owns its output row
/// exclusively, so the read-modify-write is race-free. // ponytail: still one thread per query
/// row and no shared-memory tiling of K/V — the memory term is gone, the launch geometry is not.
/// </summary>
internal static class FlashAttentionKernels
{
    /// <summary>Stats columns: max, sum-of-exp, and (backward only) Σ_d dO·O.</summary>
    public const int StatCols = 3;

    /// <summary>
    /// One thread per (b, h, i). Writes <paramref name="outp"/> row (b,i,h) and that row's
    /// softmax statistics. Rescales the running accumulator whenever a new maximum arrives,
    /// which is what keeps the single pass equal to the two-pass form.
    /// </summary>
    public static void Fwd(Index1D idx, ArrayView<float> outp, ArrayView<float> stats,
        ArrayView<float> q, ArrayView<float> k, ArrayView<float> v,
        int seqLen, int numHeads, int numKv, int headDim, float scale)
    {
        int i = idx % seqLen; int h = (idx / seqLen) % numHeads; int b = idx / (seqLen * numHeads);
        int grp = numHeads / numKv, kvh = h / grp; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        long qOff = ((long)b * seqLen + i) * qDim + (long)h * headDim;

        for (int d = 0; d < headDim; d++) outp[qOff + d] = 0f;
        float m = float.NegativeInfinity, l = 0f;
        for (int j = 0; j <= i; j++)
        {
            long kvOff = ((long)b * seqLen + j) * kvDim + (long)kvh * headDim;
            float s = 0f;
            for (int d = 0; d < headDim; d++) s += q[qOff + d] * k[kvOff + d];
            s *= scale;
            if (s > m)
            {
                // First term: m is -inf and nothing is accumulated yet, so the rescale is a
                // no-op. Guarded explicitly rather than left to exp(-inf - s), which is 0 only
                // while s is finite.
                float alpha = m == float.NegativeInfinity ? 0f : XMath.Exp(m - s);
                l *= alpha;
                for (int d = 0; d < headDim; d++) outp[qOff + d] *= alpha;
                m = s;
            }
            float p = XMath.Exp(s - m);
            l += p;
            for (int d = 0; d < headDim; d++) outp[qOff + d] += p * v[kvOff + d];
        }
        float inv = 1f / l;
        for (int d = 0; d < headDim; d++) outp[qOff + d] *= inv;
        long st = (((long)b * numHeads + h) * seqLen + i) * StatCols;
        stats[st] = m; stats[st + 1] = l;
    }

    /// <summary>
    /// Positioned, KV-length-forward variant of <see cref="Fwd"/> for inference, where the
    /// key/value matrix lives in a growable cache rather than sharing the query's batch×seqLen
    /// shape. One thread per (h, i): query rows Q[0, qLen) at absolute positions
    /// [pos0, pos0 + qLen) attend over the contiguous K/V rows K[0, kvLen) (kvLen = pos0 + qLen)
    /// with the causal mask <c>j &lt;= pos0 + i</c>. Single batch only.
    ///
    /// Decode: <c>pos0 = c</c>, <c>qLen = 1</c>, <c>kvLen = c + 1</c> — the one fresh query row at
    /// position c attends the whole cache [0, c+1]. Continued prefill after a warm cache uses the
    /// same call with <c>qLen = p</c>. K and V rows come from the cache tensors <c>k</c>/<c>v</c>
    /// [kvLen, numKv·headDim]; Q has its own rows [qLen, numHeads·headDim]. The cache row for a
    /// given position equals <c>row - pos0</c> in k/v. No probabilities are materialised.
    /// </summary>
    public static void FwdKvLen(Index1D idx, ArrayView<float> outp, ArrayView<float> stats,
        ArrayView<float> q, ArrayView<float> k, ArrayView<float> v,
        int pos0, int qLen, int kvLen, int numHeads, int numKv, int headDim, float scale)
    {
        int i = idx % qLen; int h = (idx / qLen) % numHeads;
        int grp = numHeads / numKv, kvh = h / grp; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        int absPos = pos0 + i;
        long qOff = (long)i * qDim + (long)h * headDim;

        for (int d = 0; d < headDim; d++) outp[qOff + d] = 0f;
        float m = float.NegativeInfinity, l = 0f;
        for (int j = 0; j <= absPos && j < kvLen; j++)
        {
            long kvOff = (long)j * kvDim + (long)kvh * headDim;
            float s = 0f;
            for (int d = 0; d < headDim; d++) s += q[qOff + d] * k[kvOff + d];
            s *= scale;
            if (s > m)
            {
                float alpha = m == float.NegativeInfinity ? 0f : XMath.Exp(m - s);
                l *= alpha;
                for (int d = 0; d < headDim; d++) outp[qOff + d] *= alpha;
                m = s;
            }
            float p = XMath.Exp(s - m);
            l += p;
            for (int d = 0; d < headDim; d++) outp[qOff + d] += p * v[kvOff + d];
        }
        float inv = 1f / l;
        for (int d = 0; d < headDim; d++) outp[qOff + d] *= inv;
        long st = ((long)h * qLen + i) * StatCols;
        stats[st] = m; stats[st + 1] = l;
    }

    /// <summary>
    /// D_i = Σ_d dO[i,d]·O[i,d] into stats column 2, one thread per (b, h, i). The row constant
    /// that turns dP into dS; <see cref="AttentionKernels.BwdScores"/> forms the same quantity
    /// as <c>Σ_j dP·P</c> while it still has P to hand.
    /// </summary>
    public static void BwdRowDot(Index1D idx, ArrayView<float> stats, ArrayView<float> dOut, ArrayView<float> outp,
        int seqLen, int numHeads, int headDim)
    {
        int i = idx % seqLen; int h = (idx / seqLen) % numHeads; int b = idx / (seqLen * numHeads);
        int qDim = numHeads * headDim;
        long oOff = ((long)b * seqLen + i) * qDim + (long)h * headDim;
        float acc = 0f;
        for (int d = 0; d < headDim; d++) acc += dOut[oOff + d] * outp[oOff + d];
        stats[((((long)b * numHeads + h) * seqLen + i) * StatCols) + 2] = acc;
    }

    /// <summary>dQ_i = scale · Σ_{j≤i} dS[i,j]·K_j, recomputing s_ij and p_ij per term.</summary>
    public static void BwdQ(Index1D idx, ArrayView<float> dQ, ArrayView<float> dOut,
        ArrayView<float> q, ArrayView<float> k, ArrayView<float> v, ArrayView<float> stats,
        int seqLen, int numHeads, int numKv, int headDim, float scale)
    {
        int i = idx % seqLen; int h = (idx / seqLen) % numHeads; int b = idx / (seqLen * numHeads);
        int grp = numHeads / numKv, kvh = h / grp; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        long qOff = ((long)b * seqLen + i) * qDim + (long)h * headDim;
        long st = (((long)b * numHeads + h) * seqLen + i) * StatCols;
        float m = stats[st], invL = 1f / stats[st + 1], rowD = stats[st + 2];

        for (int d = 0; d < headDim; d++) dQ[qOff + d] = 0f;
        for (int j = 0; j <= i; j++)
        {
            long kvOff = ((long)b * seqLen + j) * kvDim + (long)kvh * headDim;
            float s = 0f;
            for (int d = 0; d < headDim; d++) s += q[qOff + d] * k[kvOff + d];
            s *= scale;
            float p = XMath.Exp(s - m) * invL;
            float dp = 0f;
            for (int d = 0; d < headDim; d++) dp += dOut[qOff + d] * v[kvOff + d];
            float ds = p * (dp - rowD);
            for (int d = 0; d < headDim; d++) dQ[qOff + d] += ds * k[kvOff + d];
        }
        for (int d = 0; d < headDim; d++) dQ[qOff + d] *= scale;
    }

    /// <summary>
    /// dK_j = scale · Σ_{h∈group} Σ_{i≥j} dS[i,j]·Q_i and dV_j = Σ_{h∈group} Σ_{i≥j} P[i,j]·dO_i,
    /// one thread per (b, kvHead, j) owning the whole GQA group sum — the same ownership
    /// <see cref="AttentionKernels.BwdKV"/> uses.
    ///
    /// Unlike the materialised backward, this reads k and v inside the loop that writes dK and
    /// dV, so dK/dV may not alias k/v here. The caller enforces it.
    /// </summary>
    public static void BwdKV(Index1D idx, ArrayView<float> dK, ArrayView<float> dV, ArrayView<float> dOut,
        ArrayView<float> q, ArrayView<float> k, ArrayView<float> v, ArrayView<float> stats,
        int seqLen, int numHeads, int numKv, int headDim, float scale)
    {
        int j = idx % seqLen; int kvh = (idx / seqLen) % numKv; int b = idx / (seqLen * numKv);
        int grp = numHeads / numKv; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        long kvOff = ((long)b * seqLen + j) * kvDim + (long)kvh * headDim;

        for (int d = 0; d < headDim; d++) { dK[kvOff + d] = 0f; dV[kvOff + d] = 0f; }
        for (int g = 0; g < grp; g++)
        {
            int h = kvh * grp + g;
            for (int i = j; i < seqLen; i++)
            {
                long qOff = ((long)b * seqLen + i) * qDim + (long)h * headDim;
                long st = (((long)b * numHeads + h) * seqLen + i) * StatCols;
                float s = 0f;
                for (int d = 0; d < headDim; d++) s += q[qOff + d] * k[kvOff + d];
                s *= scale;
                float p = XMath.Exp(s - stats[st]) / stats[st + 1];
                float dp = 0f;
                for (int d = 0; d < headDim; d++) dp += dOut[qOff + d] * v[kvOff + d];
                float ds = p * (dp - stats[st + 2]);
                for (int d = 0; d < headDim; d++)
                {
                    dK[kvOff + d] += ds * q[qOff + d];
                    dV[kvOff + d] += p * dOut[qOff + d];
                }
            }
        }
        for (int d = 0; d < headDim; d++) dK[kvOff + d] *= scale;
    }
}
