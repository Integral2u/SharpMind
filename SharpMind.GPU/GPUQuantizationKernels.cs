using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using ILGPU;
using ILGPU.Runtime;
using JigSawDotNet;
using SharpMind.Core;
using SharpMind.Core.Quantization;

namespace SharpMind.GPU;

public static class GPUQuantizationKernels
{
    private const int QK = 32;
    private const int QK_K = 256;

    private static float HalfToFloatGPU(ushort h)
    {
        int sign = (h >> 15) & 0x1;
        int exp = (h >> 10) & 0x1F;
        int mant = h & 0x3FF;
        if (exp == 0)
            return sign == 0 ? mant * 5.9604644775390625e-8f : -mant * 5.9604644775390625e-8f;
        if (exp == 31)
            return mant == 0 ? (sign == 0 ? float.PositiveInfinity : float.NegativeInfinity) : float.NaN;
        int intVal = 1024 + mant;
        int shift = exp - 25;
        float result = shift >= 0 ? intVal * (1 << shift) : intVal / (float)(1 << -shift);
        return sign == 0 ? result : -result;
    }

    
    // Simple-block VecDot GPU (QK=32)
    

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ8_0), QuantizationConfig.KeyVecDotQ8_0, GPUSharpMindConfig.ValVecDotQ8_0)]
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

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ4_0), QuantizationConfig.KeyVecDotQ4_0, GPUSharpMindConfig.ValVecDotQ4_0)]
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

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ4_1), QuantizationConfig.KeyVecDotQ4_1, GPUSharpMindConfig.ValVecDotQ4_1)]
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

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ5_0), QuantizationConfig.KeyVecDotQ5_0, GPUSharpMindConfig.ValVecDotQ5_0)]
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
                int j = i / 2;
                int nib = (j % 2 == 0) ? (weights[bo + 6 + j] & 0x0F) : (weights[bo + 6 + j] >> 4);
                int q = nib | (((int)(qh >> i) & 1) << 4);
                sum += input[b * QK + i] * ((q - 16) * d);
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ5_1), QuantizationConfig.KeyVecDotQ5_1, GPUSharpMindConfig.ValVecDotQ5_1)]
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
                int j = i / 2;
                int nib = (j % 2 == 0) ? (weights[bo + 8 + j] & 0x0F) : (weights[bo + 8 + j] >> 4);
                int q = nib | (((int)(qh >> i) & 1) << 4);
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        partials[tid] = (float)sum;
    }

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ8_1), QuantizationConfig.KeyVecDotQ8_1, GPUSharpMindConfig.ValVecDotQ8_1)]
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
    

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ2K), QuantizationConfig.KeyVecDotQ2K, GPUSharpMindConfig.ValVecDotQ2K)]
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

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ4K), QuantizationConfig.KeyVecDotQ4K, GPUSharpMindConfig.ValVecDotQ4K)]
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

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ5K), QuantizationConfig.KeyVecDotQ5K, GPUSharpMindConfig.ValVecDotQ5K)]
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

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ6K), QuantizationConfig.KeyVecDotQ6K, GPUSharpMindConfig.ValVecDotQ6K)]
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

    [PuzzlePeice(nameof(QuantizationOps.VecDotQ8K), QuantizationConfig.KeyVecDotQ8K, GPUSharpMindConfig.ValVecDotQ8K)]
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

    
    // Helpers — GPU versions call CPU scalar fallbacks
    

    [PuzzlePeice(nameof(QuantizationOps.HSum256), QuantizationConfig.KeyHSum256, GPUSharpMindConfig.ValHSum256)]
    public static float HSum256_GPU(System.Runtime.Intrinsics.Vector256<float> v)
    {
        return v.GetElement(0) + v.GetElement(1) + v.GetElement(2) + v.GetElement(3)
             + v.GetElement(4) + v.GetElement(5) + v.GetElement(6) + v.GetElement(7);
    }

    [PuzzlePeice(nameof(QuantizationOps.HalfToFloat), QuantizationConfig.KeyHalfToFloat, GPUSharpMindConfig.ValHalfToFloat)]
    public static float HalfToFloat_GPU(ushort half)
    {
        return HalfToFloatGPU(half);
    }

    private static unsafe byte GetScaleMinK4_ScaleImpl(int j, byte* scales)
    {
        if (j < 4) return (byte)(scales[j] & 0x3F);
        return (byte)((scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4));
    }

    private static unsafe byte GetScaleMinK4_MinImpl(int j, byte* scales)
    {
        if (j < 4) return (byte)(scales[j + 4] & 0x3F);
        return (byte)((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4));
    }

    [PuzzlePeice(nameof(QuantizationOps.GetScaleMinK4_Scale), QuantizationConfig.KeyGetScaleMinK4_Scale, GPUSharpMindConfig.ValGetScaleMinK4_Scale)]
    public static unsafe byte GetScaleMinK4_Scale_GPU(int j, byte* scales)
    {
        return GetScaleMinK4_ScaleImpl(j, scales);
    }

    [PuzzlePeice(nameof(QuantizationOps.GetScaleMinK4_Min), QuantizationConfig.KeyGetScaleMinK4_Min, GPUSharpMindConfig.ValGetScaleMinK4_Min)]
    public static unsafe byte GetScaleMinK4_Min_GPU(int j, byte* scales)
    {
        return GetScaleMinK4_MinImpl(j, scales);
    }
}
