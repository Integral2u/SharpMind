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

    internal static unsafe void ReLUAVX2(ReadOnlySpan<float> src, Span<float> dst)
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

    internal static void ReLUScalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i] < 0f ? 0f : src[i];
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Fast transcendental helpers — polynomial approximations
    // ═══════════════════════════════════════════════════════════════════════

    /// <summary>exp(x) via range-reduced degree-6 polynomial, ≈5 ULP over [-88, 88].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> FastExp(Vector256<float> x)
    {
        x = Avx.Min(Avx.Max(x, Vector256.Create(-88.0f)), Vector256.Create(88.0f));

        // exp(x) = 2^(x * log2(e))
        var z = Avx.Multiply(x, Vector256.Create(1.4426950408889634f));

        // Round z to nearest int via magic-bias trick
        var magic = Vector256.Create(12582912.0f);
        var nF = Avx.Subtract(Avx.Add(z, magic), magic);
        var nI = Avx2.ConvertToVector256Int32(nF);

        // r = z - n  in [-0.5, 0.5]
        var r = Avx.Subtract(z, nF);

        // u = r * ln(2)  in [-0.35, 0.35]
        var u = Avx.Multiply(r, Vector256.Create(0.6931471805599453f));

        // exp(u) Horner degree-6 — error << 1 ULP on this domain
        var p = Avx.Add(Vector256.Create(1.0f),
            Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f),
                Avx.Multiply(u, Avx.Add(Vector256.Create(0.5f),
                    Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 6.0f),
                        Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 24.0f),
                            Avx.Multiply(u, Avx.Add(Vector256.Create(1.0f / 120.0f),
                                Avx.Multiply(u, Vector256.Create(1.0f / 720.0f))
                            ))
                        ))
                    ))
                ))
            ))
        );

        // Multiply by 2^n: build 2^n as a float, then multiply
        var expAdj = Avx2.Add(nI, Vector256.Create(127));
        expAdj = Avx2.Min(Avx2.Max(expAdj, Vector256.Create(0)), Vector256.Create(254));
        var pow2nBits = Avx2.ShiftLeftLogical(expAdj, 23);
        return Avx.Multiply(p, Vector256.AsSingle(pow2nBits));
    }

    /// <summary>tanh(z) = (exp(2z) - 1) / (exp(2z) + 1).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> FastTanh(Vector256<float> z)
    {
        z = Avx.Min(Avx.Max(z, Vector256.Create(-9.0f)), Vector256.Create(9.0f));
        var twoZ = Avx.Multiply(z, Vector256.Create(2.0f));
        var e2z = FastExp(twoZ);
        var one = Vector256.Create(1.0f);
        return Avx.Divide(Avx.Subtract(e2z, one), Avx.Add(e2z, one));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // GELU  0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe void GELUAVX2(ReadOnlySpan<float> src, Span<float> dst)
    {
        fixed (float* pS = src, pD = dst)
        {
            int i = 0, n = dst.Length;
            var vHalf = Vector256.Create(0.5f);
            var vSqrt2PiInv = Vector256.Create(0.7978845608f);
            var vCoeff = Vector256.Create(0.044715f);
            var one = Vector256.Create(1.0f);

            for (; i <= n - 8; i += 8)
            {
                var x = Vector256.LoadUnsafe(ref pS[i]);
                var x3 = Avx.Multiply(Avx.Multiply(x, x), x);
                var z = Avx.Multiply(vSqrt2PiInv, Avx.Add(x, Avx.Multiply(vCoeff, x3)));
                var t = FastTanh(z);
                var gelu = Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(one, t)));
                Vector256.StoreUnsafe(gelu, ref pD[i]);
            }
            for (; i < n; i++)
            {
                float x = pS[i];
                float x3 = x * x * x;
                float z = SqrtTwoPiInv * (x + GeluCoeff * x3);
                pD[i] = 0.5f * x * (1f + MathF.Tanh(z));
            }
        }
    }

    internal static void GELUScalar(ReadOnlySpan<float> src, Span<float> dst)
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
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe void SiLUAVX2(ReadOnlySpan<float> src, Span<float> dst)
    {
        fixed (float* pS = src, pD = dst)
        {
            int i = 0, n = dst.Length;
            var one = Vector256.Create(1.0f);

            for (; i <= n - 8; i += 8)
            {
                var x = Vector256.LoadUnsafe(ref pS[i]);
                var e = FastExp(Avx.Subtract(Vector256<float>.Zero, x));
                Vector256.StoreUnsafe(Avx.Multiply(x, Avx.Divide(one, Avx.Add(one, e))), ref pD[i]);
            }
            for (; i < n; i++)
                pD[i] = pS[i] / (1f + MathF.Exp(-pS[i]));
        }
    }

    internal static void SiLUScalar(ReadOnlySpan<float> src, Span<float> dst)
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

    internal static unsafe void SwiGLUAVX2(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        fixed (float* pG = gate, pU = up, pD = dst)
        {
            int i = 0, n = dst.Length;
            var one = Vector256.Create(1.0f);

            for (; i <= n - 8; i += 8)
            {
                var g = Vector256.LoadUnsafe(ref pG[i]);
                var u = Vector256.LoadUnsafe(ref pU[i]);
                var e = FastExp(Avx.Subtract(Vector256<float>.Zero, g));
                var sig = Avx.Divide(one, Avx.Add(one, e));
                Vector256.StoreUnsafe(Avx.Multiply(Avx.Multiply(g, sig), u), ref pD[i]);
            }
            for (; i < n; i++)
                pD[i] = (pG[i] / (1f + MathF.Exp(-pG[i]))) * pU[i];
        }
    }

    internal static void SwiGLUScalar(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
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

    internal static unsafe void GeGLUAVX2(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        fixed (float* pG = gate, pU = up, pD = dst)
        {
            int i = 0, n = dst.Length;
            var vHalf = Vector256.Create(0.5f);
            var vSqrt2PiInv = Vector256.Create(0.7978845608f);
            var vCoeff = Vector256.Create(0.044715f);
            var one = Vector256.Create(1.0f);

            for (; i <= n - 8; i += 8)
            {
                var g = Vector256.LoadUnsafe(ref pG[i]);
                var u = Vector256.LoadUnsafe(ref pU[i]);
                var g3 = Avx.Multiply(Avx.Multiply(g, g), g);
                var z = Avx.Multiply(vSqrt2PiInv, Avx.Add(g, Avx.Multiply(vCoeff, g3)));
                var t = FastTanh(z);
                var geluG = Avx.Multiply(vHalf, Avx.Multiply(g, Avx.Add(one, t)));
                Vector256.StoreUnsafe(Avx.Multiply(geluG, u), ref pD[i]);
            }
            for (; i < n; i++)
            {
                float g = pG[i];
                float g3 = g * g * g;
                float z = SqrtTwoPiInv * (g + GeluCoeff * g3);
                float gelu = 0.5f * g * (1f + MathF.Tanh(z));
                pD[i] = gelu * pU[i];
            }
        }
    }

    internal static void GeGLUScalar(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
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

    internal static void SoftmaxRowScalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        if (src.Length < 256)
        {
            SoftmaxRowScalarSmall(src, dst);
            return;
        }
        int n = src.Length;
        float max = src[0];
        for (int i = 1; i < n; i++) if (src[i] > max) max = src[i];

        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            dst[i] = MathF.Exp(src[i] - max);
            sum += dst[i];
        }

        float inv = 1f / sum;
        for (int i = 0; i < n; i++) dst[i] *= inv;
    }

    private static void SoftmaxRowScalarSmall(ReadOnlySpan<float> src, Span<float> dst)
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

    internal static unsafe void RMSNormRowAVX2(
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

    internal static void RMSNormRowScalar(
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

    internal static unsafe void MatMulInnerFMA(float* a, float* bt, float* c, int M, int K, int N)
    {
        const int NR = 8;
        if (M <= 1)
        {
            if (M <= 0) return;
            int nBlocks = N / NR;
            if (nBlocks > 0)
            {
                System.Threading.Tasks.Parallel.For(0, nBlocks, block =>
                {
                    int j0 = block * NR;
                    var acc0 = Vector256<float>.Zero;
                    var acc1 = Vector256<float>.Zero;
                    var acc2 = Vector256<float>.Zero;
                    var acc3 = Vector256<float>.Zero;
                    var acc4 = Vector256<float>.Zero;
                    var acc5 = Vector256<float>.Zero;
                    var acc6 = Vector256<float>.Zero;
                    var acc7 = Vector256<float>.Zero;
                    float* pBT0 = bt + (long)j0 * K;
                    float* pBT1 = pBT0 + K;
                    float* pBT2 = pBT1 + K;
                    float* pBT3 = pBT2 + K;
                    float* pBT4 = pBT3 + K;
                    float* pBT5 = pBT4 + K;
                    float* pBT6 = pBT5 + K;
                    float* pBT7 = pBT6 + K;
                    int k = 0;
                    for (; k <= K - 8; k += 8)
                    {
                        var a_vec = Vector256.LoadUnsafe(ref a[k]);
                        acc0 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT0[k]), acc0);
                        acc1 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT1[k]), acc1);
                        acc2 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT2[k]), acc2);
                        acc3 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT3[k]), acc3);
                        acc4 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT4[k]), acc4);
                        acc5 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT5[k]), acc5);
                        acc6 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT6[k]), acc6);
                        acc7 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT7[k]), acc7);
                    }
                    float s0 = MathHelpers.HSum256_Avx(acc0);
                    float s1 = MathHelpers.HSum256_Avx(acc1);
                    float s2 = MathHelpers.HSum256_Avx(acc2);
                    float s3 = MathHelpers.HSum256_Avx(acc3);
                    float s4 = MathHelpers.HSum256_Avx(acc4);
                    float s5 = MathHelpers.HSum256_Avx(acc5);
                    float s6 = MathHelpers.HSum256_Avx(acc6);
                    float s7 = MathHelpers.HSum256_Avx(acc7);
                    for (; k < K; k++)
                    {
                        float a_val = a[k];
                        s0 += a_val * pBT0[k]; s1 += a_val * pBT1[k];
                        s2 += a_val * pBT2[k]; s3 += a_val * pBT3[k];
                        s4 += a_val * pBT4[k]; s5 += a_val * pBT5[k];
                        s6 += a_val * pBT6[k]; s7 += a_val * pBT7[k];
                    }
                    c[j0] = s0; c[j0 + 1] = s1; c[j0 + 2] = s2; c[j0 + 3] = s3;
                    c[j0 + 4] = s4; c[j0 + 5] = s5; c[j0 + 6] = s6; c[j0 + 7] = s7;
                });
            }
            int tailStart = nBlocks * NR;
            for (int j = tailStart; j < N; j++)
            {
                float* pBT = bt + (long)j * K;
                var acc = Vector256<float>.Zero;
                int k = 0;
                for (; k <= K - 8; k += 8)
                    acc = Fma.MultiplyAdd(
                        Vector256.LoadUnsafe(ref a[k]),
                        Vector256.LoadUnsafe(ref pBT[k]),
                        acc);
                float sum = MathHelpers.HSum256_Avx(acc);
                for (; k < K; k++) sum += a[k] * pBT[k];
                c[j] = sum;
            }
            return;
        }
        System.Threading.Tasks.Parallel.For(0, M, i =>
        {
            float* rowA = a + (long)i * K;
            float* rowC = c + (long)i * N;
            int nBlocks = N / NR;
            for (int block = 0; block < nBlocks; block++)
            {
                int j0 = block * NR;
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;
                var acc4 = Vector256<float>.Zero;
                var acc5 = Vector256<float>.Zero;
                var acc6 = Vector256<float>.Zero;
                var acc7 = Vector256<float>.Zero;
                float* pBT0 = bt + (long)j0 * K;
                float* pBT1 = pBT0 + K;
                float* pBT2 = pBT1 + K;
                float* pBT3 = pBT2 + K;
                float* pBT4 = pBT3 + K;
                float* pBT5 = pBT4 + K;
                float* pBT6 = pBT5 + K;
                float* pBT7 = pBT6 + K;
                int k = 0;
                for (; k <= K - 8; k += 8)
                {
                    var a_vec = Vector256.LoadUnsafe(ref rowA[k]);
                    acc0 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT0[k]), acc0);
                    acc1 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT1[k]), acc1);
                    acc2 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT2[k]), acc2);
                    acc3 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT3[k]), acc3);
                    acc4 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT4[k]), acc4);
                    acc5 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT5[k]), acc5);
                    acc6 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT6[k]), acc6);
                    acc7 = Fma.MultiplyAdd(a_vec, Vector256.LoadUnsafe(ref pBT7[k]), acc7);
                }
                float s0 = MathHelpers.HSum256_Avx(acc0);
                float s1 = MathHelpers.HSum256_Avx(acc1);
                float s2 = MathHelpers.HSum256_Avx(acc2);
                float s3 = MathHelpers.HSum256_Avx(acc3);
                float s4 = MathHelpers.HSum256_Avx(acc4);
                float s5 = MathHelpers.HSum256_Avx(acc5);
                float s6 = MathHelpers.HSum256_Avx(acc6);
                float s7 = MathHelpers.HSum256_Avx(acc7);
                for (; k < K; k++)
                {
                    float a_val = rowA[k];
                    s0 += a_val * pBT0[k]; s1 += a_val * pBT1[k];
                    s2 += a_val * pBT2[k]; s3 += a_val * pBT3[k];
                    s4 += a_val * pBT4[k]; s5 += a_val * pBT5[k];
                    s6 += a_val * pBT6[k]; s7 += a_val * pBT7[k];
                }
                rowC[j0] = s0; rowC[j0 + 1] = s1; rowC[j0 + 2] = s2; rowC[j0 + 3] = s3;
                rowC[j0 + 4] = s4; rowC[j0 + 5] = s5; rowC[j0 + 6] = s6; rowC[j0 + 7] = s7;
            }
            int colTail = nBlocks * NR;
            for (int j = colTail; j < N; j++)
            {
                float* pBT = bt + (long)j * K;
                var acc = Vector256<float>.Zero;
                int k = 0;
                for (; k <= K - 8; k += 8)
                    acc = Fma.MultiplyAdd(
                        Vector256.LoadUnsafe(ref rowA[k]),
                        Vector256.LoadUnsafe(ref pBT[k]),
                        acc);
                float sum = MathHelpers.HSum256_Avx(acc);
                for (; k < K; k++) sum += rowA[k] * pBT[k];
                rowC[j] = sum;
            }
        });
    }

    internal static unsafe void MatMulInnerAVX2(float* a, float* bt, float* c, int M, int K, int N)
    {
        const int NR = 8;
        if (M <= 1)
        {
            if (M <= 0) return;
            int nBlocks = N / NR;
            if (nBlocks > 0)
            {
                System.Threading.Tasks.Parallel.For(0, nBlocks, block =>
                {
                    int j0 = block * NR;
                    var acc0 = Vector256<float>.Zero;
                    var acc1 = Vector256<float>.Zero;
                    var acc2 = Vector256<float>.Zero;
                    var acc3 = Vector256<float>.Zero;
                    var acc4 = Vector256<float>.Zero;
                    var acc5 = Vector256<float>.Zero;
                    var acc6 = Vector256<float>.Zero;
                    var acc7 = Vector256<float>.Zero;
                    float* pBT0 = bt + (long)j0 * K;
                    float* pBT1 = pBT0 + K;
                    float* pBT2 = pBT1 + K;
                    float* pBT3 = pBT2 + K;
                    float* pBT4 = pBT3 + K;
                    float* pBT5 = pBT4 + K;
                    float* pBT6 = pBT5 + K;
                    float* pBT7 = pBT6 + K;
                    int k = 0;
                    for (; k <= K - 8; k += 8)
                    {
                        var a_vec = Vector256.LoadUnsafe(ref a[k]);
                        acc0 = Avx.Add(acc0, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT0[k])));
                        acc1 = Avx.Add(acc1, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT1[k])));
                        acc2 = Avx.Add(acc2, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT2[k])));
                        acc3 = Avx.Add(acc3, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT3[k])));
                        acc4 = Avx.Add(acc4, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT4[k])));
                        acc5 = Avx.Add(acc5, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT5[k])));
                        acc6 = Avx.Add(acc6, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT6[k])));
                        acc7 = Avx.Add(acc7, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT7[k])));
                    }
                    float s0 = MathHelpers.HSum256_Avx(acc0);
                    float s1 = MathHelpers.HSum256_Avx(acc1);
                    float s2 = MathHelpers.HSum256_Avx(acc2);
                    float s3 = MathHelpers.HSum256_Avx(acc3);
                    float s4 = MathHelpers.HSum256_Avx(acc4);
                    float s5 = MathHelpers.HSum256_Avx(acc5);
                    float s6 = MathHelpers.HSum256_Avx(acc6);
                    float s7 = MathHelpers.HSum256_Avx(acc7);
                    for (; k < K; k++)
                    {
                        float a_val = a[k];
                        s0 += a_val * pBT0[k]; s1 += a_val * pBT1[k];
                        s2 += a_val * pBT2[k]; s3 += a_val * pBT3[k];
                        s4 += a_val * pBT4[k]; s5 += a_val * pBT5[k];
                        s6 += a_val * pBT6[k]; s7 += a_val * pBT7[k];
                    }
                    c[j0] = s0; c[j0 + 1] = s1; c[j0 + 2] = s2; c[j0 + 3] = s3;
                    c[j0 + 4] = s4; c[j0 + 5] = s5; c[j0 + 6] = s6; c[j0 + 7] = s7;
                });
            }
            int tailStart = nBlocks * NR;
            for (int j = tailStart; j < N; j++)
            {
                float* pBT = bt + (long)j * K;
                var acc = Vector256<float>.Zero;
                int k = 0;
                for (; k <= K - 8; k += 8)
                    acc = Avx.Add(acc, Avx.Multiply(
                        Vector256.LoadUnsafe(ref a[k]),
                        Vector256.LoadUnsafe(ref pBT[k])));
                float sum = MathHelpers.HSum256_Avx(acc);
                for (; k < K; k++) sum += a[k] * pBT[k];
                c[j] = sum;
            }
            return;
        }
        System.Threading.Tasks.Parallel.For(0, M, i =>
        {
            float* rowA = a + (long)i * K;
            float* rowC = c + (long)i * N;
            int nBlocks = N / NR;
            for (int block = 0; block < nBlocks; block++)
            {
                int j0 = block * NR;
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;
                var acc4 = Vector256<float>.Zero;
                var acc5 = Vector256<float>.Zero;
                var acc6 = Vector256<float>.Zero;
                var acc7 = Vector256<float>.Zero;
                float* pBT0 = bt + (long)j0 * K;
                float* pBT1 = pBT0 + K;
                float* pBT2 = pBT1 + K;
                float* pBT3 = pBT2 + K;
                float* pBT4 = pBT3 + K;
                float* pBT5 = pBT4 + K;
                float* pBT6 = pBT5 + K;
                float* pBT7 = pBT6 + K;
                int k = 0;
                for (; k <= K - 8; k += 8)
                {
                    var a_vec = Vector256.LoadUnsafe(ref rowA[k]);
                    acc0 = Avx.Add(acc0, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT0[k])));
                    acc1 = Avx.Add(acc1, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT1[k])));
                    acc2 = Avx.Add(acc2, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT2[k])));
                    acc3 = Avx.Add(acc3, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT3[k])));
                    acc4 = Avx.Add(acc4, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT4[k])));
                    acc5 = Avx.Add(acc5, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT5[k])));
                    acc6 = Avx.Add(acc6, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT6[k])));
                    acc7 = Avx.Add(acc7, Avx.Multiply(a_vec, Vector256.LoadUnsafe(ref pBT7[k])));
                }
                float s0 = MathHelpers.HSum256_Avx(acc0);
                float s1 = MathHelpers.HSum256_Avx(acc1);
                float s2 = MathHelpers.HSum256_Avx(acc2);
                float s3 = MathHelpers.HSum256_Avx(acc3);
                float s4 = MathHelpers.HSum256_Avx(acc4);
                float s5 = MathHelpers.HSum256_Avx(acc5);
                float s6 = MathHelpers.HSum256_Avx(acc6);
                float s7 = MathHelpers.HSum256_Avx(acc7);
                for (; k < K; k++)
                {
                    float a_val = rowA[k];
                    s0 += a_val * pBT0[k]; s1 += a_val * pBT1[k];
                    s2 += a_val * pBT2[k]; s3 += a_val * pBT3[k];
                    s4 += a_val * pBT4[k]; s5 += a_val * pBT5[k];
                    s6 += a_val * pBT6[k]; s7 += a_val * pBT7[k];
                }
                rowC[j0] = s0; rowC[j0 + 1] = s1; rowC[j0 + 2] = s2; rowC[j0 + 3] = s3;
                rowC[j0 + 4] = s4; rowC[j0 + 5] = s5; rowC[j0 + 6] = s6; rowC[j0 + 7] = s7;
            }
            int colTail = nBlocks * NR;
            for (int j = colTail; j < N; j++)
            {
                float* pBT = bt + (long)j * K;
                var acc = Vector256<float>.Zero;
                int k = 0;
                for (; k <= K - 8; k += 8)
                    acc = Avx.Add(acc, Avx.Multiply(
                        Vector256.LoadUnsafe(ref rowA[k]),
                        Vector256.LoadUnsafe(ref pBT[k])));
                float sum = MathHelpers.HSum256_Avx(acc);
                for (; k < K; k++) sum += rowA[k] * pBT[k];
                rowC[j] = sum;
            }
        });
    }

    internal static unsafe void MatMulInnerScalar(float* a, float* bt, float* c, int M, int K, int N)
    {
        if (M <= 1)
        {
            if (M <= 0) return;
            System.Threading.Tasks.Parallel.For(0, N, j =>
            {
                float* pBT = bt + (long)j * K;
                float sum = 0f;
                for (int k = 0; k < K; k++) sum += a[k] * pBT[k];
                c[j] = sum;
            });
            return;
        }
        System.Threading.Tasks.Parallel.For(0, M, i =>
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
        });
    }
}
