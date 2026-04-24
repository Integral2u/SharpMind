using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Activations;

/// <summary>
/// Pure static kernel implementations. Every method is a single unconditional
/// path — no Avx2.IsSupported checks, no branching of any kind. The factory
/// selects which method to forward to at assembly time via JigSawDotNet;
/// after that the assembled type calls the chosen kernel directly.
/// </summary>
internal static class ActivationKernels
{
    private const float SqrtTwoPiInv = 0.7978845608f;
    private const float GeluCoeff    = 0.044715f;

    // ═══════════════════════════════════════════════════════════════════════
    // ReLU
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe void ReLU_AVX2(ReadOnlySpan<float> src, Span<float> dst)
    {
        fixed (float* pS = src, pD = dst)
        {
            var zero = Vector256<float>.Zero;
            int i = 0, n = dst.Length;
            for (; i <= n - 8; i += 8)
                Vector256.StoreUnsafe(Avx.Max(zero, Vector256.LoadUnsafe(ref pS[i])), ref pD[i]);
            for (; i < n; i++)
                pD[i] = pS[i] < 0f ? 0f : pS[i];
        }
    }

    internal static void ReLU_Scalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i] < 0f ? 0f : src[i];
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GELU  (tanh approximation — no meaningful AVX2 path due to Math.Tanh)
    // 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    // ═══════════════════════════════════════════════════════════════════════

    internal static void GELU_Scalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
        {
            float x  = src[i];
            float x3 = x * x * x;
            dst[i] = 0.5f * x * (1f + MathF.Tanh(SqrtTwoPiInv * (x + GeluCoeff * x3)));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SiLU  x * sigmoid(x) = x / (1 + exp(-x))
    // No AVX2 exp path in .NET intrinsics — JIT auto-vectorises the scalar loop
    // ═══════════════════════════════════════════════════════════════════════

    internal static void SiLU_Scalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
        {
            float x = src[i];
            dst[i] = x / (1f + MathF.Exp(-x));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SwiGLU  silu(gate) * up
    // ═══════════════════════════════════════════════════════════════════════

    internal static void SwiGLU_Scalar(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            float g = gate[i];
            dst[i] = (g / (1f + MathF.Exp(-g))) * up[i];
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GeGLU  gelu(gate) * up
    // ═══════════════════════════════════════════════════════════════════════

    internal static void GeGLU_Scalar(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            float g  = gate[i];
            float g3 = g * g * g;
            float geluG = 0.5f * g * (1f + MathF.Tanh(SqrtTwoPiInv * (g + GeluCoeff * g3)));
            dst[i] = geluG * up[i];
        }
    }

    // Pass-through for gate=none
    internal static void CopyGate(ReadOnlySpan<float> gate, ReadOnlySpan<float> _, Span<float> dst)
        => gate.CopyTo(dst);

    // ═══════════════════════════════════════════════════════════════════════
    // Softmax  (numerically stable)
    // exp is the bottleneck — no benefit from AVX2 here
    // ═══════════════════════════════════════════════════════════════════════

    internal static void SoftmaxRow_Scalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        float max = src[0];
        for (int i = 1; i < src.Length; i++) if (src[i] > max) max = src[i];

        float sum = 0f;
        for (int i = 0; i < src.Length; i++) { dst[i] = MathF.Exp(src[i] - max); sum += dst[i]; }

        float inv = 1f / sum;
        for (int i = 0; i < dst.Length; i++) dst[i] *= inv;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // RMSNorm row  out[i] = src[i] * rmsInv * weight[i]
    // rmsInv is pre-computed by the Tensor-level wrapper — not computed here
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe void RMSNormRow_AVX2(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
    {
        fixed (float* pS = src, pW = weight, pD = dst)
        {
            var vRms = Vector256.Create(rmsInv);
            int i = 0, n = dst.Length;
            for (; i <= n - 8; i += 8)
                Vector256.StoreUnsafe(
                    Vector256.LoadUnsafe(ref pS[i]) * vRms * Vector256.LoadUnsafe(ref pW[i]),
                    ref pD[i]);
            for (; i < n; i++)
                pD[i] = pS[i] * rmsInv * pW[i];
        }
    }

    internal static void RMSNormRow_Scalar(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
    {
        for (int i = 0; i < dst.Length; i++)
            dst[i] = src[i] * rmsInv * weight[i];
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MatMul inner  A[M,K] × BT[N,K] → C[M,N]  (B already transposed)
    //
    // Three unconditional paths — no capability checks inside any of them:
    //   FMA    → Fma.MultiplyAdd   best throughput on Haswell+ (fused multiply-add)
    //   AVX2   → Avx.Multiply + Avx.Add  for CPUs with AVX2 but no FMA instruction set
    //   Scalar → portable fallback, JIT auto-vectorises this loop on most platforms
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe void MatMulInner_FMA(float* a, float* bt, float* c, int M, int K, int N)
    {
        for (int i = 0; i < M; i++)
        {
            float* rowA = a + (long)i * K;
            float* rowC = c + (long)i * N;
            for (int j = 0; j < N; j++)
            {
                float* rowBT = bt + (long)j * K;
                var acc = Vector256<float>.Zero;
                int k = 0;
                for (; k <= K - 8; k += 8)
                    acc = Fma.MultiplyAdd(
                        Vector256.LoadUnsafe(ref rowA[k]),
                        Vector256.LoadUnsafe(ref rowBT[k]),
                        acc);
                float sum = HSum256(acc);
                for (; k < K; k++) sum += rowA[k] * rowBT[k];
                rowC[j] = sum;
            }
        }
    }

    internal static unsafe void MatMulInner_AVX2(float* a, float* bt, float* c, int M, int K, int N)
    {
        for (int i = 0; i < M; i++)
        {
            float* rowA = a + (long)i * K;
            float* rowC = c + (long)i * N;
            for (int j = 0; j < N; j++)
            {
                float* rowBT = bt + (long)j * K;
                var acc = Vector256<float>.Zero;
                int k = 0;
                for (; k <= K - 8; k += 8)
                    acc = Avx.Add(acc, Avx.Multiply(
                        Vector256.LoadUnsafe(ref rowA[k]),
                        Vector256.LoadUnsafe(ref rowBT[k])));
                float sum = HSum256(acc);
                for (; k < K; k++) sum += rowA[k] * rowBT[k];
                rowC[j] = sum;
            }
        }
    }

    internal static unsafe void MatMulInner_Scalar(float* a, float* bt, float* c, int M, int K, int N)
    {
        for (int i = 0; i < M; i++)
        {
            float* rowA = a + (long)i * K;
            float* rowC = c + (long)i * N;
            for (int j = 0; j < N; j++)
            {
                float* rowBT = bt + (long)j * K;
                float sum = 0f;
                for (int k = 0; k < K; k++) sum += rowA[k] * rowBT[k];
                rowC[j] = sum;
            }
        }
    }
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
