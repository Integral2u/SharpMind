using ILGPU;
using ILGPU.Runtime;
using JigSawDotNet;
using SharpMind.Core.Activations;

namespace SharpMind.GPU;

public static class GPUActivationKernels
{
    private const float SqrtTwoPiInv = 0.7978845608f;
    private const float GeluCoeff = 0.044715f;

    private static Accelerator? _accelerator;
    private static Context? _context;

    private static Accelerator SharedAccelerator
    {
        get
        {
            if (_accelerator != null) return _accelerator;
            _context = Context.CreateDefault();
            _accelerator = _context.GetPreferredDevice(!GPUMode.Cuda.Equals(GPUSharpMindConfig.BestBackend)).CreateAccelerator(_context);
            return _accelerator;
        }
    }

    [PuzzlePeice(nameof(ActivationOps.ApplyPointwise), GPUSharpMindConfig.MapActivationKeyPointWise, GPUSharpMindConfig.MapActivationKernelReLU)]
    public static void ReLUGPU(ReadOnlySpan<float> src, Span<float> dst)
    {
        var acc = SharedAccelerator;
        using var bufSrc = acc.Allocate1D(src.ToArray());
        using var bufDst = acc.Allocate1D<float>(dst.Length);
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(ReLUKernel);
        kernel(dst.Length, bufDst.View, bufSrc.View);
        acc.Synchronize();
        var gpuData = bufDst.GetAsArray1D();
        for (int i = 0; i < dst.Length; i++) dst[i] = gpuData[i];
    }

    private static void ReLUKernel(Index1D index, ArrayView<float> output, ArrayView<float> input)
    {
        output[index] = input[index] < 0f ? 0f : input[index];
    }

    [PuzzlePeice(nameof(ActivationOps.ApplyPointwise), GPUSharpMindConfig.MapActivationKeyPointWise, GPUSharpMindConfig.MapActivationKernelGELU)]
    public static void GELUGPU(ReadOnlySpan<float> src, Span<float> dst)
    {
        var acc = SharedAccelerator;
        using var bufSrc = acc.Allocate1D(src.ToArray());
        using var bufDst = acc.Allocate1D<float>(dst.Length);
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, float, float>(GELUKernel);
        kernel(dst.Length, bufDst.View, bufSrc.View, SqrtTwoPiInv, GeluCoeff);
        acc.Synchronize();
        var gpuData = bufDst.GetAsArray1D();
        for (int i = 0; i < dst.Length; i++) dst[i] = gpuData[i];
    }

    private static void GELUKernel(Index1D index, ArrayView<float> output, ArrayView<float> input, float a, float c)
    {
        float x = input[index];
        float x3 = x * x * x;
        output[index] = 0.5f * x * (1f + FastTanh(a * (x + c * x3)));
    }

    private static float FastTanh(float z)
    {
        float e2z = MathF.Exp(MathF.Max(-16f, MathF.Min(16f, 2f * z)));
        return (e2z - 1f) / (e2z + 1f);
    }

    [PuzzlePeice(nameof(ActivationOps.ApplyPointwise), GPUSharpMindConfig.MapActivationKeyPointWise, GPUSharpMindConfig.MapActivationKernelSiLU)]
    public static void SiLUGPU(ReadOnlySpan<float> src, Span<float> dst)
    {
        var acc = SharedAccelerator;
        using var bufSrc = acc.Allocate1D(src.ToArray());
        using var bufDst = acc.Allocate1D<float>(dst.Length);
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SiLUKernel);
        kernel(dst.Length, bufDst.View, bufSrc.View);
        acc.Synchronize();
        var gpuData = bufDst.GetAsArray1D();
        for (int i = 0; i < dst.Length; i++) dst[i] = gpuData[i];
    }

    private static void SiLUKernel(Index1D index, ArrayView<float> output, ArrayView<float> input)
    {
        float x = input[index];
        output[index] = x / (1f + MathF.Exp(-x));
    }

    [PuzzlePeice(nameof(ActivationOps.ApplyGate), GPUSharpMindConfig.MapActivationKeyGate, GPUSharpMindConfig.MapActivationKernelSwiGLU)]
    public static void SwiGLUGPU(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        var acc = SharedAccelerator;
        using var bufGate = acc.Allocate1D(gate.ToArray());
        using var bufUp = acc.Allocate1D(up.ToArray());
        using var bufDst = acc.Allocate1D<float>(dst.Length);
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>(SwiGLUKernel);
        kernel(dst.Length, bufDst.View, bufGate.View, bufUp.View);
        acc.Synchronize();
        var gpuData = bufDst.GetAsArray1D();
        for (int i = 0; i < dst.Length; i++) dst[i] = gpuData[i];
    }

    private static void SwiGLUKernel(Index1D index, ArrayView<float> output, ArrayView<float> gate, ArrayView<float> up)
    {
        float g = gate[index];
        output[index] = g / (1f + MathF.Exp(-g)) * up[index];
    }

    [PuzzlePeice(nameof(ActivationOps.ApplyGate), GPUSharpMindConfig.MapActivationKeyGate, GPUSharpMindConfig.MapActivationKernelGeGLU)]
    public static void GeGLUGPU(ReadOnlySpan<float> gate, ReadOnlySpan<float> up, Span<float> dst)
    {
        var acc = SharedAccelerator;
        using var bufGate = acc.Allocate1D(gate.ToArray());
        using var bufUp = acc.Allocate1D(up.ToArray());
        using var bufDst = acc.Allocate1D<float>(dst.Length);
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float, float>(GeGLUKernel);
        kernel(dst.Length, bufDst.View, bufGate.View, bufUp.View, SqrtTwoPiInv, GeluCoeff);
        acc.Synchronize();
        var gpuData = bufDst.GetAsArray1D();
        for (int i = 0; i < dst.Length; i++) dst[i] = gpuData[i];
    }

    private static void GeGLUKernel(Index1D index, ArrayView<float> output, ArrayView<float> gate, ArrayView<float> up, float a, float c)
    {
        float g = gate[index];
        float g3 = g * g * g;
        float geluG = 0.5f * g * (1f + FastTanh(a * (g + c * g3)));
        output[index] = geluG * up[index];
    }

    [PuzzlePeice(nameof(ActivationOps.ApplySoftmaxRow), GPUSharpMindConfig.MapActivationKeyGate, GPUSharpMindConfig.MapActivationKernelSoftMax)]
    public static void SoftmaxRowGPU(ReadOnlySpan<float> src, Span<float> dst)
    {
        var acc = SharedAccelerator;
        using var bufSrc = acc.Allocate1D(src.ToArray());
        using var bufDst = acc.Allocate1D<float>(dst.Length);
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>(SoftmaxKernel);
        kernel(dst.Length, bufDst.View, bufSrc.View);
        acc.Synchronize();
        var gpuData = bufDst.GetAsArray1D();
        for (int i = 0; i < dst.Length; i++) dst[i] = gpuData[i];
    }

    private static void SoftmaxKernel(Index1D index, ArrayView<float> output, ArrayView<float> input)
    {
        output[index] = MathF.Exp(input[index]);
    }

    [PuzzlePeice(nameof(ActivationOps.ApplyRMSNormRow), GPUSharpMindConfig.MapActivationKeyGate, GPUSharpMindConfig.MapActivationKernelRMSNorm)]
    public static void RMSNormRowGPU(ReadOnlySpan<float> src, ReadOnlySpan<float> weight, Span<float> dst, float rmsInv)
    {
        var acc = SharedAccelerator;
        using var bufSrc = acc.Allocate1D(src.ToArray());
        using var bufW = acc.Allocate1D(weight.ToArray());
        using var bufDst = acc.Allocate1D<float>(dst.Length);
        var kernel = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, float>(RMSNormKernel);
        kernel(dst.Length, bufDst.View, bufSrc.View, bufW.View, rmsInv);
        acc.Synchronize();
        var gpuData = bufDst.GetAsArray1D();
        for (int i = 0; i < dst.Length; i++) dst[i] = gpuData[i];
    }

    private static void RMSNormKernel(Index1D index, ArrayView<float> output, ArrayView<float> src, ArrayView<float> weight, float rmsInv)
    {
        output[index] = src[index] * rmsInv * weight[index];
    }

    public static unsafe void MatMulGPU(float* a, float* bt, float* c, int M, int K, int N)
    {
        var acc = SharedAccelerator;
        var aArr = new float[M * K];
        var btArr = new float[N * K];
        for (int i = 0; i < M * K; i++) aArr[i] = a[i];
        for (int i = 0; i < N * K; i++) btArr[i] = bt[i];
        using var bufA = acc.Allocate1D<float>(M * K);
        using var bufBT = acc.Allocate1D<float>(N * K);
        using var bufC = acc.Allocate1D<float>(M * N);
        bufA.CopyFromCPU(aArr);
        bufBT.CopyFromCPU(btArr);
        var kernel = acc.LoadAutoGroupedStreamKernel<Index2D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(MatMulKernel);
        kernel(new Index2D(M, N), bufC.View, bufA.View, bufBT.View, M, K, N);
        acc.Synchronize();
        var gpuData = bufC.GetAsArray1D();
        for (int i = 0; i < M * N; i++) c[i] = gpuData[i];
    }

    private static void MatMulKernel(Index2D index, ArrayView<float> c, ArrayView<float> a, ArrayView<float> bt, int M, int K, int N)
    {
        int i = index.X;
        int j = index.Y;
        if (i >= M || j >= N) return;
        float sum = 0f;
        for (int k = 0; k < K; k++)
            sum += a[i * K + k] * bt[j * K + k];
        c[i * N + j] = sum;
    }

    public static void Synchronize() => SharedAccelerator.Synchronize();
}