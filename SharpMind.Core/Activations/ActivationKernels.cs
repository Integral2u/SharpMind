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
public static class ActivationKernels
{
    // ReLU  

    public static unsafe void ReLUAVX2(ReadOnlySpan<float> src, Span<float> dst)
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

    public static void ReLUScalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
            dst[i] = src[i] < 0f ? 0f : src[i];
    }

    
    // GELU  0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    

    public static unsafe void GELUAVX2(ReadOnlySpan<float> src, Span<float> dst)
    {
        fixed (float* pS = src, pD = dst)
        {
            int i = 0, n = dst.Length;
            var vHalf = Vector256.Create(0.5f);
            var vSqrt2PiInv = Vector256.Create(MathEx.SqrtTwoPiInv);
            var vCoeff = Vector256.Create(MathEx.GeluCoeff);
            var one = Vector256.Create(1.0f);

            for (; i <= n - 8; i += 8)
            {
                var x = Vector256.LoadUnsafe(ref pS[i]);
                var x3 = Avx.Multiply(Avx.Multiply(x, x), x);
                var z = Avx.Multiply(vSqrt2PiInv, Avx.Add(x, Avx.Multiply(vCoeff, x3)));
                var t = MathEx.FastTanh(z);
                var gelu = Avx.Multiply(vHalf, Avx.Multiply(x, Avx.Add(one, t)));
                Vector256.StoreUnsafe(gelu, ref pD[i]);
            }
            for (; i < n; i++)
                pD[i] = MathEx.Gelu(pS[i]);
        }
    }

    public static void GELUScalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
            dst[i] = MathEx.Gelu(src[i]);
    }

    
    // SiLU  x * sigmoid(x) = x / (1 + exp(-x))
    

    public static unsafe void SiLUAVX2(ReadOnlySpan<float> src, Span<float> dst)
    {
        fixed (float* pS = src, pD = dst)
        {
            int i = 0, n = dst.Length;
            var one = Vector256.Create(1.0f);

            for (; i <= n - 8; i += 8)
            {
                var x = Vector256.LoadUnsafe(ref pS[i]);
                var e = MathEx.FastExp(Avx.Subtract(Vector256<float>.Zero, x));
                Vector256.StoreUnsafe(Avx.Multiply(x, Avx.Divide(one, Avx.Add(one, e))), ref pD[i]);
            }
            for (; i < n; i++)
                pD[i] = pS[i] / (1f + MathF.Exp(-pS[i]));
        }
    }

    public static void SiLUScalar(ReadOnlySpan<float> src, Span<float> dst)
    {
        for (int i = 0; i < src.Length; i++)
        {
            float x = src[i];
            dst[i] = x / (1f + MathF.Exp(-x));
        }
    }

    
    // SwiGLU  silu(gate) * up
    

    public static unsafe void SwiGLUAVX2(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        fixed (float* pG = gate, pU = up, pD = dst)
        {
            int i = 0, n = dst.Length;
            var one = Vector256.Create(1.0f);

            for (; i <= n - 8; i += 8)
            {
                var g = Vector256.LoadUnsafe(ref pG[i]);
                var u = Vector256.LoadUnsafe(ref pU[i]);
                var e = MathEx.FastExp(Avx.Subtract(Vector256<float>.Zero, g));
                var sig = Avx.Divide(one, Avx.Add(one, e));
                Vector256.StoreUnsafe(Avx.Multiply(Avx.Multiply(g, sig), u), ref pD[i]);
            }
            for (; i < n; i++)
                pD[i] = (pG[i] / (1f + MathF.Exp(-pG[i]))) * pU[i];
        }
    }

    public static void SwiGLUScalar(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        for (int i = 0; i < dst.Length; i++)
        {
            float g = gate[i];
            dst[i] = (g / (1f + MathF.Exp(-g))) * up[i];
        }
    }

    
    // GeGLU  gelu(gate) * up
    

    public static unsafe void GeGLUAVX2(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        fixed (float* pG = gate, pU = up, pD = dst)
        {
            int i = 0, n = dst.Length;
            var vHalf = Vector256.Create(0.5f);
            var vSqrt2PiInv = Vector256.Create(MathEx.SqrtTwoPiInv);
            var vCoeff = Vector256.Create(MathEx.GeluCoeff);
            var one = Vector256.Create(1.0f);

            for (; i <= n - 8; i += 8)
            {
                var g = Vector256.LoadUnsafe(ref pG[i]);
                var u = Vector256.LoadUnsafe(ref pU[i]);
                var g3 = Avx.Multiply(Avx.Multiply(g, g), g);
                var z = Avx.Multiply(vSqrt2PiInv, Avx.Add(g, Avx.Multiply(vCoeff, g3)));
                var t = MathEx.FastTanh(z);
                var geluG = Avx.Multiply(vHalf, Avx.Multiply(g, Avx.Add(one, t)));
                Vector256.StoreUnsafe(Avx.Multiply(geluG, u), ref pD[i]);
            }
            for (; i < n; i++)
                pD[i] = MathEx.Gelu(pG[i]) * pU[i];
        }
    }

    public static void GeGLUScalar(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        for (int i = 0; i < dst.Length; i++)
            dst[i] = MathEx.Gelu(gate[i]) * up[i];
    }

    // Pass-through for gate=none
    public static void CopyGate(ReadOnlySpan<float> gate, ReadOnlySpan<float> _, Span<float> dst)
        => gate.CopyTo(dst);

    
    // Softmax  (numerically stable)
    

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void SoftmaxRowAVX2(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length;
        if (n < 8)
        {
            SoftmaxRowScalarSmall(src, dst);
            return;
        }

        fixed (float* pS = src, pD = dst)
        {
            // Pass 1: find max
            float max = pS[0];
            int i = 0;
            var vMax = Vector256.Create(pS[0]);
            for (; i <= n - 8; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref pS[i]);
                vMax = Avx.Max(vMax, v);
            }
            max = MathHelpers.HMax256_Avx(vMax);
            for (; i < n; i++)
                if (pS[i] > max) max = pS[i];

            // Pass 2: exp(x - max) and sum
            float sum = 0f;
            var vSum = Vector256<float>.Zero;
            var vMax256 = Vector256.Create(max);
            i = 0;
            for (; i <= n - 8; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref pS[i]);
                var shifted = Avx.Subtract(v, vMax256);
                var e = MathEx.FastExp(shifted);
                Vector256.StoreUnsafe(e, ref pD[i]);
                vSum = Avx.Add(vSum, e);
            }
            sum = MathHelpers.HSum256_Avx(vSum);
            for (; i < n; i++)
            {
                float e = MathF.Exp(pS[i] - max);
                pD[i] = e;
                sum += e;
            }

            // Pass 3: normalize
            float inv = 1f / sum;
            var vInv = Vector256.Create(inv);
            i = 0;
            for (; i <= n - 8; i += 8)
            {
                var e = Vector256.LoadUnsafe(ref pD[i]);
                Vector256.StoreUnsafe(Avx.Multiply(e, vInv), ref pD[i]);
            }
            for (; i < n; i++)
                pD[i] *= inv;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SoftmaxRowScalar(ReadOnlySpan<float> src, Span<float> dst)
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

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void SoftmaxRowScalarSmall(ReadOnlySpan<float> src, Span<float> dst)
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
    

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe void RMSNormRowAVX2(
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

    public static void RMSNormRowScalar(
        ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
    {
        for (int i = 0; i < dst.Length; i++)
            dst[i] = src[i] * rmsInv * weight[i];
    }

}
