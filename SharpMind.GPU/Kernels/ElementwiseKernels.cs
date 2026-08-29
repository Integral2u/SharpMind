using ILGPU;

namespace SharpMind.GPU.Kernels;

internal static class ElementwiseKernels
{
    public static void AddInPlace(Index1D i, ArrayView<float> dst, ArrayView<float> src) => dst[i] += src[i];
    public static void Copy(Index1D i, ArrayView<float> dst, ArrayView<float> src) => dst[i] = src[i];
    public static void AddBiasRows(Index1D i, ArrayView<float> x, ArrayView<float> bias, int cols) => x[i] += bias[i % cols];
    public static void Scale(Index1D i, ArrayView<float> x, float s) => x[i] *= s;
    public static void EmbedGather(Index1D i, ArrayView<float> x, ArrayView<float> table, ArrayView<int> ids, int cols)
    {
        int t = i / cols, d = i % cols;
        x[i] = table[(long)ids[t] * cols + d];
    }
}
