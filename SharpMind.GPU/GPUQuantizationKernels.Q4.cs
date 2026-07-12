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
    [PuzzlePeice(QuantizationKeys.KeyVecDotQ4_0, QuantizationKeys.KeyVecDotQ4_0, GPUSharpMindConfig.ValVecDotQ4_0)]
    public static unsafe float VecDotQ4_0_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
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
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<float>, int, int>(VecDotQ4_0Kernel);
        k(nt, bufIn.View, bufW.View, bufD.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ4_0Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> weights, ArrayView<float> dValues, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK - 1) / QK;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 18;
            float d = dValues[b];
            int be = Math.Min(QK, size - b * QK);
            for (int i = 0; i < be; i++)
            {
                int q = (weights[bo + 2 + i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * ((q - 8) * d);
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(QuantizationKeys.KeyVecDotQ4_1, QuantizationKeys.KeyVecDotQ4_1, GPUSharpMindConfig.ValVecDotQ4_1)]
    public static unsafe float VecDotQ4_1_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        var acc = GPUActivationKernels.SharedAccelerator;
        var inArr = new float[inFeatures];
        Marshal.Copy((nint)input, inArr, 0, inFeatures);
        int wOff = col * nBlocks * BLOCK_BYTES;
        var wArr = new byte[nBlocks * BLOCK_BYTES];
        Marshal.Copy((nint)(rawWeights + wOff), wArr, 0, wArr.Length);
        var dArr = new float[nBlocks];
        var mArr = new float[nBlocks];
        for (int b = 0; b < nBlocks; b++)
        {
            dArr[b] = HalfToFloatGPU((ushort)(wArr[b * BLOCK_BYTES] | (wArr[b * BLOCK_BYTES + 1] << 8)));
            mArr[b] = HalfToFloatGPU((ushort)(wArr[b * BLOCK_BYTES + 2] | (wArr[b * BLOCK_BYTES + 3] << 8)));
        }
        const int nt = 256;
        using var bufIn = acc.Allocate1D(inArr);
        using var bufW = acc.Allocate1D(wArr);
        using var bufD = acc.Allocate1D(dArr);
        using var bufM = acc.Allocate1D(mArr);
        using var bufP = acc.Allocate1D<float>(nt);
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(VecDotQ4_1Kernel);
        k(nt, bufIn.View, bufW.View, bufD.View, bufM.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ4_1Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> weights, ArrayView<float> dValues, ArrayView<float> mValues, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK - 1) / QK;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 20;
            float d = dValues[b];
            float m = mValues[b];
            int be = Math.Min(QK, size - b * QK);
            for (int i = 0; i < be; i++)
            {
                int q = (weights[bo + 4 + i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(QuantizationKeys.KeyVecDotQ4K, QuantizationKeys.KeyVecDotQ4K, GPUSharpMindConfig.ValVecDotQ4K)]
    public static unsafe float VecDotQ4K_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        var acc = GPUActivationKernels.SharedAccelerator;
        var inArr = new float[inFeatures];
        Marshal.Copy((nint)input, inArr, 0, inFeatures);
        int wOff = col * nBlocks * BLOCK_BYTES;
        var wArr = new byte[nBlocks * BLOCK_BYTES];
        Marshal.Copy((nint)(rawWeights + wOff), wArr, 0, wArr.Length);
        const int nt = 256;
        using var bufIn = acc.Allocate1D(inArr);
        using var bufW = acc.Allocate1D(wArr);
        using var bufP = acc.Allocate1D<float>(nt);
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, int, int>(VecDotQ4K_Kernel);
        k(nt, bufIn.View, bufW.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static byte GetScaleK4(int j, ArrayView<byte> w, int scalesOff)
    {
        if (j < 4) return (byte)(w[scalesOff + j] & 0x3F);
        return (byte)((w[scalesOff + j + 4] & 0x0F) | ((w[scalesOff + j - 4] >> 6) << 4));
    }

    private static byte GetMinK4(int j, ArrayView<byte> w, int scalesOff)
    {
        if (j < 4) return (byte)(w[scalesOff + j + 4] & 0x3F);
        return (byte)((w[scalesOff + j + 4] >> 4) | ((w[scalesOff + j] >> 6) << 4));
    }

    private static void VecDotQ4K_Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> w, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 144;
            float dSuper = HalfToFloatGPU((ushort)(w[bo] | (w[bo + 1] << 8)));
            float minSuper = HalfToFloatGPU((ushort)(w[bo + 2] | (w[bo + 3] << 8)));
            int blockEnd = Math.Min(QK_K, size - b * QK_K);
            for (int j = 0; j < blockEnd; j += 64)
            {
                int idx2 = j / 64;
                int qOff = idx2 * 32;
                for (int half = 0; half < 2; half++)
                {
                    int subIdx = idx2 * 2 + half;
                    float d1 = dSuper * GetScaleK4(subIdx, w, bo + 4);
                    float m1 = minSuper * GetMinK4(subIdx, w, bo + 4);
                    int hEnd = Math.Min(32, blockEnd - j - half * 32);
                    for (int l = 0; l < hEnd; l++)
                    {
                        int q = half == 0 ? (w[bo + 16 + qOff + l] & 0x0F) : (w[bo + 16 + qOff + l] >> 4);
                        sum += input[b * QK_K + j + half * 32 + l] * (d1 * q - m1);
                    }
                }
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(QuantizationKeys.KeyQuantizedMatMulQ4_0, QuantizationKeys.KeyQuantizedMatMulQ4_0, GPUSharpMindConfig.ValQuantizedMatMulQ4_0_Serial)]
    public static unsafe void QuantizedMatMulQ4_0_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_0_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(QuantizationKeys.KeyQuantizedMatMulQ4_0, QuantizationKeys.KeyQuantizedMatMulQ4_0, GPUSharpMindConfig.ValQuantizedMatMulQ4_0_Parallel)]
    public static unsafe void QuantizedMatMulQ4_0_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_0_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_0_GPU(pInRow, rawWeights, col, K);
            });
        }
    }

    [PuzzlePeice(QuantizationKeys.KeyQuantizedMatMulQ4_1, QuantizationKeys.KeyQuantizedMatMulQ4_1, GPUSharpMindConfig.ValQuantizedMatMulQ4_1_Serial)]
    public static unsafe void QuantizedMatMulQ4_1_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_1_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(QuantizationKeys.KeyQuantizedMatMulQ4_1, QuantizationKeys.KeyQuantizedMatMulQ4_1, GPUSharpMindConfig.ValQuantizedMatMulQ4_1_Parallel)]
    public static unsafe void QuantizedMatMulQ4_1_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4_1_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_1_GPU(pInRow, rawWeights, col, K);
            });
        }
    }

    [PuzzlePeice(QuantizationKeys.KeyQuantizedMatMulQ4K, QuantizationKeys.KeyQuantizedMatMulQ4K, GPUSharpMindConfig.ValQuantizedMatMulQ4K_Serial)]
    public static unsafe void QuantizedMatMulQ4K_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4K_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(QuantizationKeys.KeyQuantizedMatMulQ4K, QuantizationKeys.KeyQuantizedMatMulQ4K, GPUSharpMindConfig.ValQuantizedMatMulQ4K_Parallel)]
    public static unsafe void QuantizedMatMulQ4K_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ4K_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4K_GPU(pInRow, rawWeights, col, K);
            });
        }
    }
}
