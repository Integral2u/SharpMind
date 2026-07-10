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
    
    // ReLU  

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

    
    // Fast transcendental helpers — polynomial approximations
    

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

    
    // GELU  0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    

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

    
    // SiLU  x * sigmoid(x) = x / (1 + exp(-x))
    

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

    
    // SwiGLU  silu(gate) * up
    

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

    
    // GeGLU  gelu(gate) * up
    

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

    
    // Softmax  (numerically stable)
    // exp is the bottleneck — no benefit from AVX2 here
    

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

    
    // RMSNorm row  out[i] = src[i] * rmsInv * weight[i]
    // rmsInv is pre-computed by the Tensor-level wrapper — not computed here
    

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

}
