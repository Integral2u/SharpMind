using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;

namespace SharpMind.Training;

// Static kernels — one pure unconditional path each

public static class TrainingKernels
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Training)}.{nameof(TrainingKernels)}";

    // AdamW per-element update — AVX2
    // m = β₁m + (1-β₁)g
    // v = β₂v + (1-β₂)g²
    // θ -= lr * (m̂/(√v̂+ε) + λθ)   (λ=0 for no-decay params)

    public static unsafe void AdamWUpdate_AVX2(
        Span<float> data, Span<float> grad,
        Span<float> m,    Span<float> v,
        float beta1,   float beta2,
        float bc1,     float bc2,
        float lr,      float epsilon,
        float decay)
    {
        var vBeta1   = Vector256.Create(beta1);
        var vBeta2   = Vector256.Create(beta2);
        var v1mBeta1 = Vector256.Create(1f - beta1);
        var v1mBeta2 = Vector256.Create(1f - beta2);
        var vLr      = Vector256.Create(lr);
        var vEps     = Vector256.Create(epsilon);
        var vDecay   = Vector256.Create(decay);
        var vBc1     = Vector256.Create(bc1);
        var vBc2     = Vector256.Create(bc2);

        fixed (float* pD = data, pG = grad, pM = m, pV = v)
        {
            int i = 0, n = data.Length;
            for (; i <= n - 8; i += 8)
            {
                var vg = Vector256.LoadUnsafe(ref pG[i]);
                var vm = Avx.Add(Avx.Multiply(vBeta1, Vector256.LoadUnsafe(ref pM[i])),
                                 Avx.Multiply(v1mBeta1, vg));
                var vv = Avx.Add(Avx.Multiply(vBeta2, Vector256.LoadUnsafe(ref pV[i])),
                                 Avx.Multiply(v1mBeta2, Avx.Multiply(vg, vg)));
                Vector256.StoreUnsafe(vm, ref pM[i]);
                Vector256.StoreUnsafe(vv, ref pV[i]);

                var mhat   = Avx.Divide(vm, vBc1);
                var vhat   = Avx.Divide(vv, vBc2);
                var vd     = Vector256.LoadUnsafe(ref pD[i]);
                var update = Avx.Multiply(vLr,
                                 Avx.Add(Avx.Divide(mhat,
                                             Avx.Add(Avx.Sqrt(vhat), vEps)),
                                         Avx.Multiply(vDecay, vd)));
                Vector256.StoreUnsafe(Avx.Subtract(vd, update), ref pD[i]);
            }

            for (; i < n; i++)
            {
                float g    = pG[i];
                pM[i] = beta1 * pM[i] + (1f - beta1) * g;
                pV[i] = beta2 * pV[i] + (1f - beta2) * g * g;
                float mhat = pM[i] / bc1;
                float vhat = pV[i] / bc2;
                pD[i] -= lr * (mhat / (MathF.Sqrt(vhat) + epsilon) + decay * pD[i]);
            }
        }
    }

    public static void AdamWUpdate_Scalar(
        Span<float> data, Span<float> grad,
        Span<float> m,    Span<float> v,
        float beta1,   float beta2,
        float bc1,     float bc2,
        float lr,      float epsilon,
        float decay)
    {
        for (int i = 0; i < data.Length; i++)
        {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1f - beta1) * g;
            v[i] = beta2 * v[i] + (1f - beta2) * g * g;
            float mhat = m[i] / bc1;
            float vhat = v[i] / bc2;
            data[i] -= lr * (mhat / (MathF.Sqrt(vhat) + epsilon) + decay * data[i]);
        }
    }

    // L2 norm accumulation — AVX2

    public static unsafe float L2NormSq_AVX2(ReadOnlySpan<float> data)
    {
        fixed (float* p = data)
        {
            var acc = Vector256<float>.Zero;
            int i   = 0, n = data.Length;
            for (; i <= n - 8; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref p[i]);
                acc = Fma.IsSupported
                    ? Fma.MultiplyAdd(v, v, acc)
                    : Avx.Add(acc, Avx.Multiply(v, v));
            }
            float sum = MathHelpers.HSum256_Avx(acc);
            for (; i < n; i++) sum += p[i] * p[i];
            return sum;
        }
    }

    public static float L2NormSq_Scalar(ReadOnlySpan<float> data)
    {
        float sum = 0f;
        foreach (float v in data) sum += v * v;
        return sum;
    }
}
