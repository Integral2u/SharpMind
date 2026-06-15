using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;

namespace SharpMind.Model.Layers;

// Static kernels — one pure unconditional path each

internal static class NormKernels
{
    // RMSNorm row — out[i] = src[i] * rmsInv * weight[i]

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

    // RMSNorm param — 1 / sqrt(mean(v²) + eps)
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

    // LayerNorm row — standard mean/variance normalisation
    // out[i] = (src[i] - mean) / sqrt(var + eps) * weight[i] + bias[i]

    internal static unsafe void LayerNormRowAVX2(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight,
        ReadOnlySpan<float> bias, Span<float> dst, float eps)
    {
        fixed (float* pS = src, pW = weight, pB = bias, pD = dst)
        {
            int i = 0, n = dst.Length;

            // Pass 1: sum → mean
            var vSum = Vector256<float>.Zero;
            for (; i <= n - 8; i += 8)
                vSum = Avx.Add(vSum, Vector256.LoadUnsafe(ref pS[i]));
            float total = MathHelpers.HSum256_Avx(vSum);
            for (; i < n; i++)
                total += pS[i];
            float mean = total / n;
            var vMean = Vector256.Create(mean);

            // Pass 2: sum of (x - mean)² → variance
            i = 0;
            var vVar = Vector256<float>.Zero;
            for (; i <= n - 8; i += 8)
            {
                var d = Avx.Subtract(Vector256.LoadUnsafe(ref pS[i]), vMean);
                vVar = Avx.Add(vVar, Avx.Multiply(d, d));
            }
            float varTotal = MathHelpers.HSum256_Avx(vVar);
            for (; i < n; i++)
            {
                float d = pS[i] - mean;
                varTotal += d * d;
            }
            float variance = varTotal / n;
            float invStd = 1f / MathF.Sqrt(variance + eps);
            var vInvStd = Vector256.Create(invStd);

            // Pass 3: output = (x - mean) * invStd * weight + bias
            i = 0;
            for (; i <= n - 8; i += 8)
            {
                var vS = Vector256.LoadUnsafe(ref pS[i]);
                var vW = Vector256.LoadUnsafe(ref pW[i]);
                var vB = Vector256.LoadUnsafe(ref pB[i]);
                var norm = Avx.Multiply(Avx.Subtract(vS, vMean), vInvStd);
                Vector256.StoreUnsafe(Avx.Add(Avx.Multiply(norm, vW), vB), ref pD[i]);
            }
            for (; i < n; i++)
            {
                float norm = (pS[i] - mean) * invStd;
                pD[i] = norm * pW[i] + pB[i];
            }
        }
    }

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
