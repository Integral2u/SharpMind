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
    [PuzzlePeice(nameof(QuantizationOps.VecDotQ5_0), QuantizationKeys.KeyVecDotQ5_0, GPUSharpMindConfig.ValVecDotQ5_0)]
    public static unsafe float VecDotQ5_0_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 22;
        int nBlocks = (inFeatures + QK - 1) / QK;
        var acc = GPUActivationKernels.SharedAccelerator;
        var inArr = new float[inFeatures];
        Marshal.Copy((nint)input, inArr, 0, inFeatures);
        int wOff = col * nBlocks * BLOCK_BYTES;
        var wArr = new byte[nBlocks * BLOCK_BYTES];
        Marshal.Copy((nint)(rawWeights + wOff), wArr, 0, wArr.Length);
        var dArr = new float[nBlocks];
        var qhArr = new uint[nBlocks];
        for (int b = 0; b < nBlocks; b++)
        {
            dArr[b] = HalfToFloatGPU((ushort)(wArr[b * BLOCK_BYTES] | (wArr[b * BLOCK_BYTES + 1] << 8)));
            qhArr[b] = (uint)(wArr[b * BLOCK_BYTES + 2] | (wArr[b * BLOCK_BYTES + 3] << 8) | (wArr[b * BLOCK_BYTES + 4] << 16) | (wArr[b * BLOCK_BYTES + 5] << 24));
        }
        const int nt = 256;
        using var bufIn = acc.Allocate1D(inArr);
        using var bufW = acc.Allocate1D(wArr);
        using var bufD = acc.Allocate1D(dArr);
        using var bufQH = acc.Allocate1D(qhArr);
        using var bufP = acc.Allocate1D<float>(nt);
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<uint>, ArrayView<float>, int, int>(VecDotQ5_0Kernel);
        k(nt, bufIn.View, bufW.View, bufD.View, bufQH.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ5_0Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> weights, ArrayView<float> dValues, ArrayView<uint> qhValues, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK - 1) / QK;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 22;
            float d = dValues[b];
            uint qh = qhValues[b];
            int be = Math.Min(QK, size - b * QK);
            for (int i = 0; i < be; i++)
            {
                int nib = ((i & 1) == 0) ? (weights[bo + 6 + i / 2] & 0x0F) : (weights[bo + 6 + i / 2] >> 4);
                int q = nib | (((int)(qh >> i) & 1) << 4);
                sum += input[b * QK + i] * ((q - 16) * d);
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ5_1), QuantizationKeys.KeyVecDotQ5_1, GPUSharpMindConfig.ValVecDotQ5_1)]
    public static unsafe float VecDotQ5_1_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 24;
        int nBlocks = (inFeatures + QK - 1) / QK;
        var acc = GPUActivationKernels.SharedAccelerator;
        var inArr = new float[inFeatures];
        Marshal.Copy((nint)input, inArr, 0, inFeatures);
        int wOff = col * nBlocks * BLOCK_BYTES;
        var wArr = new byte[nBlocks * BLOCK_BYTES];
        Marshal.Copy((nint)(rawWeights + wOff), wArr, 0, wArr.Length);
        var dArr = new float[nBlocks];
        var mArr = new float[nBlocks];
        var qhArr = new uint[nBlocks];
        for (int b = 0; b < nBlocks; b++)
        {
            dArr[b] = HalfToFloatGPU((ushort)(wArr[b * BLOCK_BYTES] | (wArr[b * BLOCK_BYTES + 1] << 8)));
            mArr[b] = HalfToFloatGPU((ushort)(wArr[b * BLOCK_BYTES + 2] | (wArr[b * BLOCK_BYTES + 3] << 8)));
            qhArr[b] = (uint)(wArr[b * BLOCK_BYTES + 4] | (wArr[b * BLOCK_BYTES + 5] << 8) | (wArr[b * BLOCK_BYTES + 6] << 16) | (wArr[b * BLOCK_BYTES + 7] << 24));
        }
        const int nt = 256;
        using var bufIn = acc.Allocate1D(inArr);
        using var bufW = acc.Allocate1D(wArr);
        using var bufD = acc.Allocate1D(dArr);
        using var bufM = acc.Allocate1D(mArr);
        using var bufQH = acc.Allocate1D(qhArr);
        using var bufP = acc.Allocate1D<float>(nt);
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, ArrayView<float>, ArrayView<uint>, ArrayView<float>, int, int>(VecDotQ5_1Kernel);
        k(nt, bufIn.View, bufW.View, bufD.View, bufM.View, bufQH.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ5_1Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> weights, ArrayView<float> dValues, ArrayView<float> mValues, ArrayView<uint> qhValues, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK - 1) / QK;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 24;
            float d = dValues[b];
            float m = mValues[b];
            uint qh = qhValues[b];
            int be = Math.Min(QK, size - b * QK);
            for (int i = 0; i < be; i++)
            {
                int nib = ((i & 1) == 0) ? (weights[bo + 8 + i / 2] & 0x0F) : (weights[bo + 8 + i / 2] >> 4);
                int q = nib | (((int)(qh >> i) & 1) << 4);
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ5K), QuantizationKeys.KeyVecDotQ5K, GPUSharpMindConfig.ValVecDotQ5K)]
    public static unsafe float VecDotQ5K_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 176;
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
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, int, int>(VecDotQ5K_Kernel);
        k(nt, bufIn.View, bufW.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ5K_Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> w, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 176;
            float d = HalfToFloatGPU((ushort)(w[bo] | (w[bo + 1] << 8)));
            float minv = HalfToFloatGPU((ushort)(w[bo + 2] | (w[bo + 3] << 8)));
            int blockEnd = Math.Min(QK_K, size - b * QK_K);
            for (int j = 0; j < blockEnd; j += 64)
            {
                int idx2 = j / 64;
                int qOff = idx2 * 32;
                int u1 = 1 << (idx2 * 2);
                int u2 = 2 << (idx2 * 2);
                for (int half = 0; half < 2; half++)
                {
                    int subIdx = idx2 * 2 + half;
                    float d1 = d * GetScaleK4(subIdx, w, bo + 4);
                    float m1 = minv * GetMinK4(subIdx, w, bo + 4);
                    int hEnd = Math.Min(32, blockEnd - j - half * 32);
                    int bit = half == 0 ? u1 : u2;
                    for (int l = 0; l < hEnd; l++)
                    {
                        int low = half == 0 ? (w[bo + 48 + qOff + l] & 0x0F) : (w[bo + 48 + qOff + l] >> 4);
                        int q = low + (((w[bo + 16 + l] & bit) != 0) ? 16 : 0);
                        sum += input[b * QK_K + j + half * 32 + l] * (d1 * q - m1);
                    }
                }
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ5_0), QuantizationKeys.KeyQuantizedMatMulQ5_0, GPUSharpMindConfig.ValQuantizedMatMulQ5_0_Serial)]
    public static unsafe void QuantizedMatMulQ5_0_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5_0_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ5_0), QuantizationKeys.KeyQuantizedMatMulQ5_0, GPUSharpMindConfig.ValQuantizedMatMulQ5_0_Parallel)]
    public static unsafe void QuantizedMatMulQ5_0_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ5_0_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_0_GPU(pInRow, rawWeights, col, K);
            });
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ5_1), QuantizationKeys.KeyQuantizedMatMulQ5_1, GPUSharpMindConfig.ValQuantizedMatMulQ5_1_Serial)]
    public static unsafe void QuantizedMatMulQ5_1_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5_1_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ5_1), QuantizationKeys.KeyQuantizedMatMulQ5_1, GPUSharpMindConfig.ValQuantizedMatMulQ5_1_Parallel)]
    public static unsafe void QuantizedMatMulQ5_1_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ5_1_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_1_GPU(pInRow, rawWeights, col, K);
            });
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ5K), QuantizationKeys.KeyQuantizedMatMulQ5K, GPUSharpMindConfig.ValQuantizedMatMulQ5K_Serial)]
    public static unsafe void QuantizedMatMulQ5K_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ5K_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ5K), QuantizationKeys.KeyQuantizedMatMulQ5K, GPUSharpMindConfig.ValQuantizedMatMulQ5K_Parallel)]
    public static unsafe void QuantizedMatMulQ5K_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ5K_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5K_GPU(pInRow, rawWeights, col, K);
            });
        }
    }
}
