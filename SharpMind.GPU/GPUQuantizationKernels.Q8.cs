using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using ILGPU;
using ILGPU.Runtime;
using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Quantization;

namespace SharpMind.GPU;

public static partial class GPUQuantizationKernels
{
    // Simple-block VecDot GPU (QK=32)

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ8_0), QuantizationKeys.KeyVecDotQ8_0, GPUSharpMindConfig.ValVecDotQ8_0)]
    public static unsafe float VecDotQ8_0_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        var acc = GPUActivationKernels.SharedAccelerator;
        var inArr = new float[inFeatures];
        Marshal.Copy((nint)input, inArr, 0, inFeatures);
        int wOff = col * nBlocks * BLOCK_BYTES;
        var wArr = new byte[nBlocks * BLOCK_BYTES];
        Marshal.Copy((nint)(rawWeights + wOff), wArr, 0, wArr.Length);
        var dArr = new float[nBlocks];
        for (int b = 0; b < nBlocks; b++)
            dArr[b] = HalfToFloatGPU((ushort)(wArr[b * BLOCK_BYTES] | (wArr[b * BLOCK_BYTES + 1] << 8)));
        const int nt = 256;
        using var bufIn = acc.Allocate1D(inArr);
        using var bufW = acc.Allocate1D(wArr);
        using var bufD = acc.Allocate1D(dArr);
        using var bufP = acc.Allocate1D<float>(nt);
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<float>, int, int>(VecDotSimpleKernel);
        k(nt, bufIn.View, bufW.View, bufD.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotSimpleKernel(Index1D idx, ArrayView<float> input, ArrayView<byte> weights, ArrayView<float> dValues, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK - 1) / QK;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 34;
            float d = dValues[b];
            int be = Math.Min(QK, size - b * QK);
            for (int i = 0; i < be; i++)
            {
                sbyte q = (sbyte)weights[bo + 2 + i];
                sum += input[b * QK + i] * (q * d);
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ8_1), QuantizationKeys.KeyVecDotQ8_1, GPUSharpMindConfig.ValVecDotQ8_1)]
    public static unsafe float VecDotQ8_1_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        var acc = GPUActivationKernels.SharedAccelerator;
        var inArr = new float[inFeatures];
        Marshal.Copy((nint)input, inArr, 0, inFeatures);
        int wOff = col * nBlocks * BLOCK_BYTES;
        var wArr = new byte[nBlocks * BLOCK_BYTES];
        Marshal.Copy((nint)(rawWeights + wOff), wArr, 0, wArr.Length);
        var dArr = new float[nBlocks];
        for (int b = 0; b < nBlocks; b++)
            dArr[b] = HalfToFloatGPU((ushort)(wArr[b * BLOCK_BYTES] | (wArr[b * BLOCK_BYTES + 1] << 8)));
        const int nt = 256;
        using var bufIn = acc.Allocate1D(inArr);
        using var bufW = acc.Allocate1D(wArr);
        using var bufD = acc.Allocate1D(dArr);
        using var bufP = acc.Allocate1D<float>(nt);
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<float>, int, int>(VecDotQ8_1Kernel);
        k(nt, bufIn.View, bufW.View, bufD.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ8_1Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> weights, ArrayView<float> dValues, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK - 1) / QK;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 36;
            float d = dValues[b];
            int be = Math.Min(QK, size - b * QK);
            for (int i = 0; i < be; i++)
            {
                sbyte q = (sbyte)weights[bo + 4 + i];
                sum += input[b * QK + i] * (q * d);
            }
        }
        partials[tid] = (float)sum;
    }

    // K-Quant VecDot GPU (QK_K=256)

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ8K), QuantizationKeys.KeyVecDotQ8K, GPUSharpMindConfig.ValVecDotQ8K)]
    public static unsafe float VecDotQ8K_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        var acc = GPUActivationKernels.SharedAccelerator;
        var inArr = new float[inFeatures];
        Marshal.Copy((nint)input, inArr, 0, inFeatures);
        int wOff = col * nBlocks * BLOCK_BYTES;
        var wArr = new byte[nBlocks * BLOCK_BYTES];
        Marshal.Copy((nint)(rawWeights + wOff), wArr, 0, wArr.Length);
        var dArr = new float[nBlocks];
        for (int b = 0; b < nBlocks; b++)
            dArr[b] = BitConverter.ToSingle(wArr, b * BLOCK_BYTES);
        const int nt = 256;
        using var bufIn = acc.Allocate1D(inArr);
        using var bufW = acc.Allocate1D(wArr);
        using var bufD = acc.Allocate1D(dArr);
        using var bufP = acc.Allocate1D<float>(nt);
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<float>, int, int>(VecDotQ8K_Kernel);
        k(nt, bufIn.View, bufW.View, bufD.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ8K_Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> weights, ArrayView<float> dValues, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 292;
            float d = dValues[b];
            int be = Math.Min(QK_K, size - b * QK_K);
            for (int i = 0; i < be; i++)
            {
                sbyte q = (sbyte)weights[bo + 4 + i];
                sum += input[b * QK_K + i] * (q * d);
            }
        }
        partials[tid] = (float)sum;
    }

    // Quantized MatMul wrappers

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ8_0), QuantizationKeys.KeyQuantizedMatMulQ8_0, GPUSharpMindConfig.ValQuantizedMatMulQ8_0_Serial)]
    public static unsafe void QuantizedMatMulQ8_0_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_0_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ8_0), QuantizationKeys.KeyQuantizedMatMulQ8_0, GPUSharpMindConfig.ValQuantizedMatMulQ8_0_Parallel)]
    public static unsafe void QuantizedMatMulQ8_0_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ8_0_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_0_GPU(pInRow, rawWeights, col, K);
            });
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ8_1), QuantizationKeys.KeyQuantizedMatMulQ8_1, GPUSharpMindConfig.ValQuantizedMatMulQ8_1_Serial)]
    public static unsafe void QuantizedMatMulQ8_1_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_1_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ8_1), QuantizationKeys.KeyQuantizedMatMulQ8_1, GPUSharpMindConfig.ValQuantizedMatMulQ8_1_Parallel)]
    public static unsafe void QuantizedMatMulQ8_1_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ8_1_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_1_GPU(pInRow, rawWeights, col, K);
            });
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ8K), QuantizationKeys.KeyQuantizedMatMulQ8K, GPUSharpMindConfig.ValQuantizedMatMulQ8K_Serial)]
    public static unsafe void QuantizedMatMulQ8K_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8K_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ8K), QuantizationKeys.KeyQuantizedMatMulQ8K, GPUSharpMindConfig.ValQuantizedMatMulQ8K_Parallel)]
    public static unsafe void QuantizedMatMulQ8K_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ8K_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8K_GPU(pInRow, rawWeights, col, K);
            });
        }
    }
}
