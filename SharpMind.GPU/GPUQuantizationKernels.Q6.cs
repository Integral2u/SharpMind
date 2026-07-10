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
    [PuzzlePeice(nameof(QuantizationOps.VecDotQ6K), QuantizationKeys.KeyVecDotQ6K, GPUSharpMindConfig.ValVecDotQ6K)]
    public static unsafe float VecDotQ6K_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 210;
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
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, int, int>(VecDotQ6K_Kernel);
        k(nt, bufIn.View, bufW.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ6K_Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> w, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 210;
            float d = HalfToFloatGPU((ushort)(w[bo + 208] | (w[bo + 209] << 8)));
            int blockEnd = Math.Min(QK_K, size - b * QK_K);
            for (int nOff = 0; nOff < QK_K; nOff += 128)
            {
                int qlOff = nOff == 0 ? 0 : 64;
                int qhOff = nOff == 0 ? 0 : 32;
                int scOff = nOff == 0 ? 0 : 8;
                for (int l = 0; l < 32 && nOff + l < blockEnd; l++)
                {
                    int q1 = (w[bo + qlOff + l] & 0x0F) | ((w[bo + 128 + qhOff + l] & 0x03) << 4);
                    int scIdx = l / 16;
                    float s1 = (sbyte)w[bo + 192 + scOff + scIdx * 2];
                    sum += input[b * QK_K + nOff + l] * (d * s1 * (q1 - 32));
                }
                for (int l = 0; l < 32 && nOff + 32 + l < blockEnd; l++)
                {
                    int q2 = (w[bo + qlOff + l + 32] & 0x0F) | (((w[bo + 128 + qhOff + l] >> 2) & 0x03) << 4);
                    int scIdx = l / 16;
                    float s2 = (sbyte)w[bo + 192 + scOff + scIdx * 2 + 1];
                    sum += input[b * QK_K + nOff + 32 + l] * (d * s2 * (q2 - 32));
                }
                for (int l = 0; l < 32 && nOff + 64 + l < blockEnd; l++)
                {
                    int q3 = ((w[bo + qlOff + l] >> 4) & 0x0F) | (((w[bo + 128 + qhOff + l] >> 4) & 0x03) << 4);
                    int scIdx = l / 16;
                    float s3 = (sbyte)w[bo + 192 + scOff + scIdx * 2];
                    sum += input[b * QK_K + nOff + 64 + l] * (d * s3 * (q3 - 32));
                }
                for (int l = 0; l < 32 && nOff + 96 + l < blockEnd; l++)
                {
                    int q4 = ((w[bo + qlOff + l + 32] >> 4) & 0x0F) | (((w[bo + 128 + qhOff + l] >> 6) & 0x03) << 4);
                    int scIdx = l / 16;
                    float s4 = (sbyte)w[bo + 192 + scOff + scIdx * 2 + 1];
                    sum += input[b * QK_K + nOff + 96 + l] * (d * s4 * (q4 - 32));
                }
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ6K), QuantizationKeys.KeyQuantizedMatMulQ6K, GPUSharpMindConfig.ValQuantizedMatMulQ6K_Serial)]
    public static unsafe void QuantizedMatMulQ6K_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ6K_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ6K), QuantizationKeys.KeyQuantizedMatMulQ6K, GPUSharpMindConfig.ValQuantizedMatMulQ6K_Parallel)]
    public static unsafe void QuantizedMatMulQ6K_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ6K_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ6K_GPU(pInRow, rawWeights, col, K);
            });
        }
    }
}
