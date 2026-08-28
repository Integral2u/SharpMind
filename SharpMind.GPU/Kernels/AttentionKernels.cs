using ILGPU;
using ILGPU.Algorithms;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// Causal attention with materialised probabilities, as BackpropEngine.ForwardAttention.
/// Forward: one thread per (b, h, i) query row. Backward in three launches so that no
/// two threads write the same element: (1) dP and dS per (b,h,i) row into scratch,
/// (2) dQ per (b,h,i), (3) dK/dV per (b, kvh, j) summing over the query heads of the
/// group and over i ≥ j. // ponytail: O(S²·D) per thread, no tiling; flash-style when S grows.
/// </summary>
internal static class AttentionKernels
{
    public static void Fwd(Index1D idx, ArrayView<float> outp, ArrayView<float> probs, ArrayView<float> q, ArrayView<float> k, ArrayView<float> v,
        int seqLen, int numHeads, int numKv, int headDim, float scale)
    {
        int i = idx % seqLen; int h = (idx / seqLen) % numHeads; int b = idx / (seqLen * numHeads);
        int grp = numHeads / numKv, kvh = h / grp; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        long qOff = ((long)b * seqLen + i) * qDim + (long)h * headDim;
        long pBase = (((long)b * numHeads + h) * seqLen + i) * seqLen;
        float max = float.NegativeInfinity;
        for (int j = 0; j <= i; j++)
        {
            long kOff = ((long)b * seqLen + j) * kvDim + (long)kvh * headDim;
            float s = 0f; for (int d = 0; d < headDim; d++) s += q[qOff + d] * k[kOff + d];
            s *= scale; probs[pBase + j] = s; if (s > max) max = s;
        }
        float sum = 0f;
        for (int j = 0; j <= i; j++) { float p = XMath.Exp(probs[pBase + j] - max); probs[pBase + j] = p; sum += p; }
        float inv = 1f / sum;
        for (int j = 0; j <= i; j++) probs[pBase + j] *= inv;
        for (int d = 0; d < headDim; d++)
        {
            float acc = 0f;
            for (int j = 0; j <= i; j++) acc += probs[pBase + j] * v[((long)b * seqLen + j) * kvDim + (long)kvh * headDim + d];
            outp[qOff + d] = acc;
        }
    }

    // (1) dS[i,j] = P[i,j]·(dP[i,j] − Σ_j dP·P), dP = dO_i · V_j  → scratch (row i of head (b,h))
    public static void BwdScores(Index1D idx, ArrayView<float> dS, ArrayView<float> dOut, ArrayView<float> v, ArrayView<float> probs,
        int seqLen, int numHeads, int numKv, int headDim)
    {
        int i = idx % seqLen; int h = (idx / seqLen) % numHeads; int b = idx / (seqLen * numHeads);
        int grp = numHeads / numKv, kvh = h / grp; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        long oOff = ((long)b * seqLen + i) * qDim + (long)h * headDim;
        long pBase = (((long)b * numHeads + h) * seqLen + i) * seqLen;
        float rowSum = 0f;
        for (int j = 0; j <= i; j++)
        {
            long vOff = ((long)b * seqLen + j) * kvDim + (long)kvh * headDim;
            float dp = 0f; for (int d = 0; d < headDim; d++) dp += dOut[oOff + d] * v[vOff + d];
            dS[pBase + j] = dp; rowSum += dp * probs[pBase + j];
        }
        for (int j = 0; j <= i; j++) dS[pBase + j] = probs[pBase + j] * (dS[pBase + j] - rowSum);
        // Nothing downstream reads j > i, but the scratch is arena memory: leaving stale
        // NaN/Inf there would poison any future full-row consumer for one cheap pass.
        for (int j = i + 1; j < seqLen; j++) dS[pBase + j] = 0f;
    }

    // (2) dQ_i = scale · Σ_j dS[i,j] · K_j
    public static void BwdQ(Index1D idx, ArrayView<float> dQ, ArrayView<float> dS, ArrayView<float> k, int seqLen, int numHeads, int numKv, int headDim, float scale)
    {
        int i = idx % seqLen; int h = (idx / seqLen) % numHeads; int b = idx / (seqLen * numHeads);
        int grp = numHeads / numKv, kvh = h / grp; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        long qOff = ((long)b * seqLen + i) * qDim + (long)h * headDim;
        long pBase = (((long)b * numHeads + h) * seqLen + i) * seqLen;
        for (int d = 0; d < headDim; d++)
        {
            float acc = 0f;
            for (int j = 0; j <= i; j++) acc += dS[pBase + j] * k[((long)b * seqLen + j) * kvDim + (long)kvh * headDim + d];
            dQ[qOff + d] = scale * acc;
        }
    }

    // (3) dK_j = scale · Σ_{h∈group} Σ_{i≥j} dS[i,j]·Q_i ;  dV_j = Σ_{h∈group} Σ_{i≥j} P[i,j]·dO_i
    // This one thread owns the whole group sum that the CPU engine builds by calling the
    // per-head backward once per (b,h) and AddHead-ing into the shared kv slot.
    public static void BwdKV(Index1D idx, ArrayView<float> dK, ArrayView<float> dV, ArrayView<float> dS, ArrayView<float> probs, ArrayView<float> q, ArrayView<float> dOut,
        int seqLen, int numHeads, int numKv, int headDim, float scale)
    {
        int j = idx % seqLen; int kvh = (idx / seqLen) % numKv; int b = idx / (seqLen * numKv);
        int grp = numHeads / numKv; int qDim = numHeads * headDim, kvDim = numKv * headDim;
        long kOff = ((long)b * seqLen + j) * kvDim + (long)kvh * headDim;
        for (int d = 0; d < headDim; d++)
        {
            float accK = 0f, accV = 0f;
            for (int g = 0; g < grp; g++)
            {
                int h = kvh * grp + g;
                for (int i = j; i < seqLen; i++)
                {
                    long pIdx = (((long)b * numHeads + h) * seqLen + i) * seqLen + j;
                    long oOff = ((long)b * seqLen + i) * qDim + (long)h * headDim + d;
                    accK += dS[pIdx] * q[oOff];
                    accV += probs[pIdx] * dOut[oOff];
                }
            }
            dK[kOff + d] = scale * accK;
            dV[kOff + d] = accV;
        }
    }
}
