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
    [PuzzlePeice(nameof(QuantizationOps.VecDotQ3K), QuantizationConfig.KeyVecDotQ3K, GPUSharpMindConfig.ValVecDotQ3K)]
    public static unsafe float VecDotQ3K_GPU(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 110;
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
        var k = acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<byte>, ArrayView<float>, int, int>(VecDotQ3K_Kernel);
        k(nt, bufIn.View, bufW.View, bufP.View, nt, inFeatures);
        acc.Synchronize();
        var ps = bufP.GetAsArray1D();
        float sum = 0f;
        for (int i = 0; i < nt; i++) sum += ps[i];
        return sum;
    }

    private static void VecDotQ3K_Kernel(Index1D idx, ArrayView<float> input, ArrayView<byte> w, ArrayView<float> partials, int nt, int size)
    {
        int tid = idx.X;
        int nB = (size + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = tid; b < nB; b += nt)
        {
            int bo = b * 110;
            float dAll = HalfToFloatGPU((ushort)(w[bo + 108] | (w[bo + 109] << 8)));
            int blockEnd = Math.Min(QK_K, size - b * QK_K);
            uint aux0 = (uint)(w[bo + 96] | (w[bo + 97] << 8) | (w[bo + 98] << 16) | (w[bo + 99] << 24));
            uint aux1 = (uint)(w[bo + 100] | (w[bo + 101] << 8) | (w[bo + 102] << 16) | (w[bo + 103] << 24));
            uint aux2 = (uint)(w[bo + 104] | (w[bo + 105] << 8) | (w[bo + 106] << 16) | (w[bo + 107] << 24));
            uint kmask1 = 0x03030303u;
            uint kmask2 = 0x0f0f0f0fu;
            uint tmp = aux2;
            uint s0 = (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
            uint s1 = (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
            uint s2 = ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            uint s3 = ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            for (int i = 0; i < blockEnd; i++)
            {
                int qsByte = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int s2v = (w[bo + 32 + qsByte] >> qsShift) & 3;
                int hBit = (w[bo + (i % 32)] >> (i / 32)) & 1;
                int actual = s2v - (hBit == 0 ? 4 : 0);
                int si = i / 16;
                int sc;
                if (si < 4) sc = (sbyte)((s0 >> (si * 8)) & 0xFF);
                else if (si < 8) sc = (sbyte)((s1 >> ((si - 4) * 8)) & 0xFF);
                else if (si < 12) sc = (sbyte)((s2 >> ((si - 8) * 8)) & 0xFF);
                else sc = (sbyte)((s3 >> ((si - 12) * 8)) & 0xFF);
                float val = dAll * (sc - 32) * actual;
                sum += input[b * QK_K + i] * val;
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ3K), QuantizationConfig.KeyQuantizedMatMulQ3K, GPUSharpMindConfig.ValQuantizedMatMulQ3K_Serial)]
    public static unsafe void QuantizedMatMulQ3K_Serial_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ3K_GPU(pInRow, rawWeights, col, K);
        }
    }

    [PuzzlePeice(nameof(QuantizationOps.QuantizedMatMulQ3K), QuantizationConfig.KeyQuantizedMatMulQ3K, GPUSharpMindConfig.ValQuantizedMatMulQ3K_Parallel)]
    public static unsafe void QuantizedMatMulQ3K_Parallel_GPU(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            for (int col = 0; col < N; col++)
                output[col] = VecDotQ3K_GPU(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ3K_GPU(pInRow, rawWeights, col, K);
            });
        }
    }
}
