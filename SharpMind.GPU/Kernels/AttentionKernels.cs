using ILGPU;
using ILGPU.Algorithms;

namespace SharpMind.GPU.Kernels;

/// <summary>
/// The two row passes of causal attention with materialised probabilities. Everything else
/// — scores, output, dP, dQ, dK, dV — is a GEMM issued by <see cref="GpuKernels.AttnFwd"/> and
/// <see cref="GpuKernels.AttnBwd"/>, so those six products run on cuBLAS instead of on a
/// hand-rolled thread-per-row loop.
///
/// Both kernels are one thread per (b, h, i) query row over S contiguous floats, which is the
/// part that genuinely is a row reduction: a softmax over the row, and its backward. The
/// upper triangle is written (not skipped) so no caller has to pre-zero the tensor — the GEMM
/// fills the whole S×S block, including the j > i scores the mask discards.
/// </summary>
internal static class AttentionKernels
{
    /// <summary>
    /// Causal softmax of one score row, in place: scale, mask, softmax over j ≤ i, zero above.
    /// <paramref name="probs"/> arrives holding the raw Q·Kᵀ scores.
    /// </summary>
    public static void SoftmaxRow(Index1D idx, ArrayView<float> probs, int seqLen, float scale)
    {
        int row = idx;
        int i = row % seqLen;
        long pBase = (long)row * seqLen;
        float max = float.NegativeInfinity;
        for (int j = 0; j <= i; j++) { float s = probs[pBase + j] * scale; probs[pBase + j] = s; if (s > max) max = s; }
        float sum = 0f;
        for (int j = 0; j <= i; j++) { float p = XMath.Exp(probs[pBase + j] - max); probs[pBase + j] = p; sum += p; }
        float inv = 1f / sum;
        for (int j = 0; j <= i; j++) probs[pBase + j] *= inv;
        for (int j = i + 1; j < seqLen; j++) probs[pBase + j] = 0f;
    }

    /// <summary>
    /// Softmax backward for one row, in place: dS = scale · P ∘ (dP − Σ_j dP·P).
    /// <paramref name="dS"/> arrives holding dP = dO·Vᵀ; the scale the scores were multiplied by
    /// is folded in here, so the dQ and dK GEMMs that read dS need no scaling of their own.
    /// </summary>
    public static void SoftmaxRowBwd(Index1D idx, ArrayView<float> dS, ArrayView<float> probs, int seqLen, float scale)
    {
        int row = idx;
        int i = row % seqLen;
        long pBase = (long)row * seqLen;
        float rowSum = 0f;
        for (int j = 0; j <= i; j++) rowSum += dS[pBase + j] * probs[pBase + j];
        for (int j = 0; j <= i; j++) dS[pBase + j] = scale * probs[pBase + j] * (dS[pBase + j] - rowSum);
        // Nothing downstream reads j > i, but the scratch is arena memory and the dK/dV GEMMs
        // read the whole S×S block: stale values there would be summed into the gradient.
        for (int j = i + 1; j < seqLen; j++) dS[pBase + j] = 0f;
    }
}
