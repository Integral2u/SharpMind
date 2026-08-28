using ILGPU;
using ILGPU.Algorithms;

namespace SharpMind.GPU.Kernels;

/// <summary>One thread per row. // ponytail: row-per-thread reductions; warp-reduce when rows get long.</summary>
internal static class NormKernels
{
    // y = x·rInv·w, rInv = 1/sqrt(mean(x²)+eps)        (NormLayer.ForwardWithState for RmsNormLayer)
    public static void RmsNormFwd(Index1D row, ArrayView<float> y, ArrayView<float> rInvOut, ArrayView<float> x, ArrayView<float> w, int d, float eps)
    {
        long b = (long)(int)row * d;
        float ss = 0f;
        for (int i = 0; i < d; i++) { float v = x[b + i]; ss += v * v; }
        float rInv = 1f / XMath.Sqrt(ss / d + eps);
        rInvOut[row] = rInv;
        for (int i = 0; i < d; i++) y[b + i] = x[b + i] * rInv * w[i];
    }

    // g = dy·w ; dx = rInv·(g − xNorm·mean(g·xNorm)), xNorm = x·rInv   (GradientMapping.RMSNorm, frozen w)
    public static void RmsNormBwd(Index1D row, ArrayView<float> dx, ArrayView<float> dy, ArrayView<float> x, ArrayView<float> rInv, ArrayView<float> w, int d)
    {
        long b = (long)(int)row * d;
        float r = rInv[row];
        float mean = 0f;
        for (int i = 0; i < d; i++) mean += dy[b + i] * w[i] * x[b + i] * r;
        mean /= d;
        for (int i = 0; i < d; i++) dx[b + i] = r * (dy[b + i] * w[i] - x[b + i] * r * mean);
    }
}
