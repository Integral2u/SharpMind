using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Model.Layers;

// ─────────────────────────────────────────────────────────────────────────────
// Fused kernels — combine multiple ops to reduce memory bandwidth
// ─────────────────────────────────────────────────────────────────────────────

internal static class FusedKernels
{
    // ── Fused RMSNorm + Linear ─────────────────────────────────────────────
    // Computes: output = (RMSNorm(src, normWeight) @ weight + bias
    //
    // Input:  src [HiddenDim], normWeight [HiddenDim], weight [HiddenDim, HiddenDim], bias [HiddenDim] or null
    // Output: dst [HiddenDim]

    internal static unsafe void FusedRMSNormLinearAVX2(
        ReadOnlySpan<float> src,
        ReadOnlySpan<float> normWeight,
        ReadOnlySpan<float> weight,
        float* bias,
        Span<float> dst,
        float rmsInv)
    {
        fixed (float* pS = src, pN = normWeight, pW = weight, pD = dst)
        {
            int n = dst.Length;
            var vRmsInv = Vector256.Create(rmsInv);

            float* pWRow = pW;
            for (int o = 0; o < n; o++)
            {
                var acc = Vector256<float>.Zero;
                int i = 0;

                for (; i <= n - 8; i += 8)
                {
                    var normed = Vector256.LoadUnsafe(ref pS[i]) * vRmsInv * Vector256.LoadUnsafe(ref pN[i]);
                    var w = Vector256.LoadUnsafe(ref pWRow[i]);
                    acc = Fma.IsSupported
                        ? Fma.MultiplyAdd(normed, w, acc)
                        : Avx.Add(acc, Avx.Multiply(normed, w));
                }

                float sum = HSum256(acc);
                for (; i < n; i++)
                    sum += pS[i] * rmsInv * pN[i] * pWRow[i];

                pD[o] = sum + (bias != null ? bias[o] : 0f);
                pWRow += n;
            }
        }
    }

    internal static unsafe void FusedRMSNormLinearScalar(
        ReadOnlySpan<float> src,
        ReadOnlySpan<float> normWeight,
        ReadOnlySpan<float> weight,
        float* bias,
        Span<float> dst,
        float rmsInv)
    {
        int n = dst.Length;
        for (int o = 0; o < n; o++)
        {
            float sum = 0f;
            for (int i = 0; i < n; i++)
            {
                float normed = src[i] * rmsInv * normWeight[i];
                sum += normed * weight[o * n + i];
            }
            dst[o] = sum + (bias != null ? bias[o] : 0f);
        }
    }

    // ── Fused RMSNorm + Linear (residual form) ───────────────────────────
    // Computes: output = residual + (RMSNorm(src) @ weight + bias)
    // Used in transformer block: h = x + Layer(Norm(x))

    internal static unsafe void FusedRMSNormLinearResidualAVX2(
        ReadOnlySpan<float> src,
        ReadOnlySpan<float> residual,
        ReadOnlySpan<float> normWeight,
        ReadOnlySpan<float> weight,
        float* bias,
        Span<float> dst,
        float rmsInv)
    {
        fixed (float* pS = src, pR = residual, pN = normWeight, pW = weight, pD = dst)
        {
            int n = dst.Length;
            var vRmsInv = Vector256.Create(rmsInv);

            float* pWRow = pW;
            for (int o = 0; o < n; o++)
            {
                var acc = Vector256<float>.Zero;
                int i = 0;

                for (; i <= n - 8; i += 8)
                {
                    var normed = Vector256.LoadUnsafe(ref pS[i]) * vRmsInv * Vector256.LoadUnsafe(ref pN[i]);
                    var w = Vector256.LoadUnsafe(ref pWRow[i]);
                    acc = Fma.IsSupported
                        ? Fma.MultiplyAdd(normed, w, acc)
                        : Avx.Add(acc, Avx.Multiply(normed, w));
                }

                float sum = HSum256(acc);
                for (; i < n; i++)
                    sum += pS[i] * rmsInv * pN[i] * pWRow[i];

                pD[o] = pR[o] + sum + (bias != null ? bias[o] : 0f);
                pWRow += n;
            }
        }
    }

    internal static unsafe void FusedRMSNormLinearResidualScalar(
        ReadOnlySpan<float> src,
        ReadOnlySpan<float> residual,
        ReadOnlySpan<float> normWeight,
        ReadOnlySpan<float> weight,
        float* bias,
        Span<float> dst,
        float rmsInv)
    {
        int n = dst.Length;
        for (int o = 0; o < n; o++)
        {
            float sum = 0f;
            for (int i = 0; i < n; i++)
            {
                float normed = src[i] * rmsInv * normWeight[i];
                sum += normed * weight[o * n + i];
            }
            dst[o] = residual[o] + sum + (bias != null ? bias[o] : 0f);
        }
    }

    // ── Helpers ────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum256(Vector256<float> v)
    {
        var lo = Avx.ExtractVector128(v, 0);
        var hi = Avx.ExtractVector128(v, 1);
        var s = Sse.Add(lo, hi);
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        return Sse.AddScalar(s, Sse.Shuffle(s, s, 1)).ToScalar();
    }
}