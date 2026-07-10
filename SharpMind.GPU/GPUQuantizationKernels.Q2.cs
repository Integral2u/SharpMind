using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Quantization;

namespace SharpMind.GPU;

public static partial class GPUQuantizationKernels
{
    [PuzzlePeice(nameof(QuantizationOps.VecDotQ2K), QuantizationKeys.KeyVecDotQ2K, GPUSharpMindConfig.ValVecDotQ2K)]
    public static unsafe float VecDotQ2K_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
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
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, int, int>(VecDotQ2K_Kernel);
        k(nt, bufIn.View, bufW.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ2K_Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> w, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 84;
            float dSuper = HalfToFloatGPU((ushort)(w[bo + 80] | (w[bo + 81] << 8)));
            float minSuper = HalfToFloatGPU((ushort)(w[bo + 82] | (w[bo + 83] << 8)));
            int blockEnd = Math.Min(QK_K, size - b * QK_K);
            int qOff = 0;
            for (int n16 = 0; n16 < QK_K; n16 += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int isc = (n16 / 128) * 8 + j * 2;
                    float dl = dSuper * (w[bo + isc] & 0x0F);
                    float ml = minSuper * (w[bo + isc] >> 4);
                    for (int l = 0; l < 16 && b * QK_K + n16 + j * 32 + l < b * QK_K + blockEnd; l++)
                    {
                        int v = (w[bo + 16 + qOff + l] >> shift) & 3;
                        sum += input[b * QK_K + n16 + j * 32 + l] * (dl * v - ml);
                    }
                    dl = dSuper * (w[bo + isc + 1] & 0x0F);
                    ml = minSuper * (w[bo + isc + 1] >> 4);
                    for (int l = 0; l < 16 && b * QK_K + n16 + j * 32 + 16 + l < b * QK_K + blockEnd; l++)
                    {
                        int v = (w[bo + 16 + qOff + l + 16] >> shift) & 3;
                        sum += input[b * QK_K + n16 + j * 32 + 16 + l] * (dl * v - ml);
                    }
                    shift += 2;
                }
                qOff += 32;
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ2K), QuantizationKeys.KeyQuantizedMatMulQ2K, GPUSharpMindConfig.ValQuantizedMatMulQ2K_Serial)]
    public static unsafe void QuantizedMatMulQ2K_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ2K_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ2K), QuantizationKeys.KeyQuantizedMatMulQ2K, GPUSharpMindConfig.ValQuantizedMatMulQ2K_Parallel)]
    public static unsafe void QuantizedMatMulQ2K_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ2K_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ2K_GPU(pInRow, rawWeights, col, K);
            });
        }
    }
}
