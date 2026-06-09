using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;

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

    // ── RMSNorm param — 1 / sqrt(mean(v²) + eps) ────────────────────────
    // Single-pass stable variant with overflow guard in the scalar fallback.
    // The AVX2 path uses direct sum-of-squares; overflow is not possible
    // for inference-range values (|v| < 10⁶).

    internal static unsafe float RMSNormParamAVX2(ReadOnlySpan<float> row, float eps)
    {
        fixed (float* pRow = row)
        {
            int i = 0, n = row.Length;
            var vSum = Vector256<float>.Zero;

            for (; i <= n - 8; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref pRow[i]);
                vSum = Avx.Add(vSum, Avx.Multiply(v, v));
            }

            float ss = MathHelpers.HSum256_Avx(vSum);

            for (; i < n; i++)
                ss += pRow[i] * pRow[i];

            return 1f / MathF.Sqrt(ss / n + eps);
        }
    }

    internal static float RMSNormParamScalar(ReadOnlySpan<float> row, float eps)
    {
        float maxAbs = 0f;
        float ss = 0f;
        int n = row.Length;

        foreach (float v in row)
        {
            float a = Math.Abs(v);
            if (a > maxAbs)
            {
                float ratio = maxAbs / a;
                ss = ss * ratio * ratio + 1f;
                maxAbs = a;
            }
            else if (a > 1e-20f)
            {
                float vn = v / maxAbs;
                ss += vn * vn;
            }
        }

        if (maxAbs < 1e-20f)
            return 1f / MathF.Sqrt(eps);

        float rms = maxAbs * MathF.Sqrt(ss / n + eps / (maxAbs * maxAbs));
        return 1f / rms;
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
