using System.Runtime.Intrinsics;

namespace SharpMind.Model.Layers;

// ─────────────────────────────────────────────────────────────────────────────
// Static kernels — one pure unconditional path each
// ─────────────────────────────────────────────────────────────────────────────

internal static class NormKernels
{
    // ── RMSNorm row — out[i] = src[i] * rmsInv * weight[i] ───────────────

    internal static unsafe void RMSNormRowAVX2(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
    {
        fixed (float* pS = src, pW = weight, pD = dst)
        {
            var vRms = Vector256.Create(rmsInv);
            int i = 0, n = dst.Length;
            for (; i <= n - 8; i += 8)
                (Vector256.LoadUnsafe(ref pS[i]) * vRms * Vector256.LoadUnsafe(ref pW[i])).StoreUnsafe(
                    ref pD[i]);
            for (; i < n; i++)
                pD[i] = pS[i] * rmsInv * pW[i];
        }
    }

    internal static void RMSNormRowScalar(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
    {
        for (int i = 0; i < dst.Length; i++)
            dst[i] = src[i] * rmsInv * weight[i];
    }

    // ── LayerNorm row — standard mean/variance normalisation ─────────────
    // out[i] = (src[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]

    internal static void LayerNormRowAVX2(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, Span<float> dst, float eps)
        => LayerNormRowScalar(src, weight, bias, dst, eps); // exp-dominated; AVX2 gives no gain

    internal static void LayerNormRowScalar(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, Span<float> dst, float eps)
    {
        float mean = 0f;
        foreach (float v in src) mean += v;
        mean /= src.Length;

        float variance = 0f;
        foreach (float v in src) { float d = v - mean; variance += d * d; }
        variance /= src.Length;

        float invStd = 1f / MathF.Sqrt(variance + eps);
        for (int i = 0; i < dst.Length; i++)
            dst[i] = (src[i] - mean) * invStd * weight[i] + bias[i];
    }
}
