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
    /// <summary>Column-parallel path for M == 1 (decode) — GPU variant.</summary>
    private static unsafe void DecodeParallelGPU(VecDotFn vecdot,
        float* input, byte* rawWeights, float* output,
        int K, int N)
    {
        const int MinColsForParallel = 512;
        if (N < MinColsForParallel)
        {
            for (int col = 0; col < N; col++)
                output[col] = vecdot(input, rawWeights, col, K);
            return;
        }

        int numChunks = Math.Min(Environment.ProcessorCount,
            (N + MinColsForParallel - 1) / MinColsForParallel);
        int chunkSize = (N + numChunks - 1) / numChunks;

        long inputAddr = (long)input;
        long weightsAddr = (long)rawWeights;
        long outputAddr = (long)output;

        System.Threading.Tasks.Parallel.For(0, numChunks, chunkIdx =>
        {
            float* pInput = (float*)inputAddr;
            byte* pWeights = (byte*)weightsAddr;
            float* pOutput = (float*)outputAddr;

            int colStart = chunkIdx * chunkSize;
            int colEnd = Math.Min(colStart + chunkSize, N);

            for (int col = colStart; col < colEnd; col++)
                pOutput[col] = vecdot(pInput, pWeights, col, K);
        });
    }
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

    [PuzzlePeice(QuantizationKeys.KeyHSum256, QuantizationKeys.KeyHSum256, GPUSharpMindConfig.ValHSum256)]
    public static float HSum256_GPU(Vector256<float> v)
    {
        return v.GetElement(0) + v.GetElement(1) + v.GetElement(2) + v.GetElement(3)
             + v.GetElement(4) + v.GetElement(5) + v.GetElement(6) + v.GetElement(7);
    }

    [PuzzlePeice(QuantizationKeys.KeyHalfToFloat, QuantizationKeys.KeyHalfToFloat, GPUSharpMindConfig.ValHalfToFloat)]
    public static float HalfToFloat_GPU(ushort half)
    {
        return HalfToFloatGPU(half);
    }

    private static ushort FloatToHalf_GPUImpl(float f)
    {
        uint bits = (uint)BitConverter.SingleToInt32Bits(f);
        uint sign = (bits >> 16) & 0x8000;
        int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
        uint mant = bits & 0x007FFFFF;
        if (exp <= 0)
        {
            if (exp < -10) return (ushort)sign;
            mant = (mant | 0x00800000) >> (1 - exp);
            return (ushort)(sign | (mant >> 13));
        }
        if (exp >= 31)
        {
            if (exp > 31) return (ushort)(sign | 0x7C00 | (mant >> 13));
            return (ushort)(sign | 0x7C00 | (mant != 0 ? 0x200u : 0));
        }
        return (ushort)(sign | ((uint)exp << 10) | (mant >> 13));
    }

    [PuzzlePeice(QuantizationKeys.KeyFloatToHalf, QuantizationKeys.KeyFloatToHalf, GPUSharpMindConfig.ValFloatToHalf)]
    public static ushort FloatToHalf_GPU(float f)
    {
        return FloatToHalf_GPUImpl(f);
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

    [PuzzlePeice(QuantizationKeys.KeyGetScaleMinK4_Scale, QuantizationKeys.KeyGetScaleMinK4_Scale, GPUSharpMindConfig.ValGetScaleMinK4_Scale)]
    public static unsafe byte GetScaleMinK4_Scale_GPU(int j, byte* scales)
    {
        return GetScaleMinK4_ScaleImpl(j, scales);
    }

    [PuzzlePeice(QuantizationKeys.KeyGetScaleMinK4_Min, QuantizationKeys.KeyGetScaleMinK4_Min, GPUSharpMindConfig.ValGetScaleMinK4_Min)]
    public static unsafe byte GetScaleMinK4_Min_GPU(int j, byte* scales)
    {
        return GetScaleMinK4_MinImpl(j, scales);
    }
}
