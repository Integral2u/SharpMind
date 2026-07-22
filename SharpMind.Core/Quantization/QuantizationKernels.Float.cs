using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Runtime.InteropServices;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{ 

    public static unsafe void QuantizedMatMulF32_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        float* w = (float*)rawWeights;
        for (int row = 0; row < M; row++)
        {
            float* pIn = input + (long)row * K;
            float* pOut = output + (long)row * N;
            for (int col = 0; col < N; col++)
            {
                float sum = 0;
                float* pW = w + (long)col * K;
                for (int i = 0; i < K; i++)
                    sum += pIn[i] * pW[i];
                pOut[col] = sum;
            }
        }
    }

    public static unsafe void QuantizedMatMulF32_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotF32_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            float* w = (float*)rawWeights;
            Parallel.For(0, M, row =>
            {
                float* pIn = input + (long)row * K;
                float* pOut = output + (long)row * N;
                for (int col = 0; col < N; col++)
                {
                    float sum = 0;
                    float* pW = w + (long)col * K;
                    for (int i = 0; i < K; i++)
                        sum += pIn[i] * pW[i];
                    pOut[col] = sum;
                }
            });
        }
    }

    public static unsafe void QuantizedMatMulF16_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        ushort* w = (ushort*)rawWeights;
        for (int row = 0; row < M; row++)
        {
            float* pIn = input + (long)row * K;
            float* pOut = output + (long)row * N;
            for (int col = 0; col < N; col++)
            {
                float sum = 0;
                ushort* pW = w + (long)col * K;
                for (int i = 0; i < K; i++)
                    sum += pIn[i] * HalfToFloat_Scalar(pW[i]);
                pOut[col] = sum;
            }
        }
    }

    public static unsafe void QuantizedMatMulF16_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotF16_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            ushort* w = (ushort*)rawWeights;
            Parallel.For(0, M, row =>
            {
                float* pIn = input + (long)row * K;
                float* pOut = output + (long)row * N;
                for (int col = 0; col < N; col++)
                {
                    float sum = 0;
                    ushort* pW = w + (long)col * K;
                    for (int i = 0; i < K; i++)
                        sum += pIn[i] * HalfToFloat_Scalar(pW[i]);
                    pOut[col] = sum;
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotF32_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        float* w = (float*)rawWeights;
        double sum = 0;
        float* pW = w + (long)col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotF16_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        ushort* w = (ushort*)rawWeights;
        double sum = 0;
        ushort* pW = w + (long)col * inFeatures;
        for (int i = 0; i < inFeatures; i++)
            sum += input[i] * HalfToFloat_Scalar(pW[i]);
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotF32_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        float* w = (float*)rawWeights;
        float* pW = w + (long)col * inFeatures;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        int i = 0;
        for (; i <= inFeatures - 16; i += 16)
        {
            var vi0 = Vector256.LoadUnsafe(ref pW[i]);
            var vi1 = Vector256.LoadUnsafe(ref pW[i + 8]);
            var vw0 = Vector256.LoadUnsafe(ref input[i]);
            var vw1 = Vector256.LoadUnsafe(ref input[i + 8]);
            vacc0 = Fma.MultiplyAdd(vi0, vw0, vacc0);
            vacc1 = Fma.MultiplyAdd(vi1, vw1, vacc1);
        }
        for (; i <= inFeatures - 8; i += 8)
        {
            var vi = Vector256.LoadUnsafe(ref pW[i]);
            var vw = Vector256.LoadUnsafe(ref input[i]);
            vacc0 = Fma.MultiplyAdd(vi, vw, vacc0);
        }
        float sum = MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        for (; i < inFeatures; i++)
            sum += input[i] * pW[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotF16_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        ushort* w = (ushort*)rawWeights;
        ushort* pW = w + (long)col * inFeatures;
        float* stackBuf = stackalloc float[8];
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        int i = 0;
        for (; i <= inFeatures - 16; i += 16)
        {
            for (int j = 0; j < 8; j++) stackBuf[j] = HalfToFloat_F16C(pW[i + j]);
            var vw0 = Vector256.Load(stackBuf);
            for (int j = 0; j < 8; j++) stackBuf[j] = HalfToFloat_F16C(pW[i + 8 + j]);
            var vw1 = Vector256.Load(stackBuf);
            var vi0 = Vector256.LoadUnsafe(ref input[i]);
            var vi1 = Vector256.LoadUnsafe(ref input[i + 8]);
            vacc0 = Fma.MultiplyAdd(vw0, vi0, vacc0);
            vacc1 = Fma.MultiplyAdd(vw1, vi1, vacc1);
        }
        for (; i <= inFeatures - 8; i += 8)
        {
            for (int j = 0; j < 8; j++) stackBuf[j] = HalfToFloat_F16C(pW[i + j]);
            var vw = Vector256.Load(stackBuf);
            var vi = Vector256.LoadUnsafe(ref input[i]);
            vacc0 = Fma.MultiplyAdd(vw, vi, vacc0);
        }
        float sum = MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        for (; i < inFeatures; i++)
            sum += input[i] * HalfToFloat_F16C(pW[i]);
        return sum;
    }

    public static unsafe void QuantizedMatMulF32_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
        => QuantizedMatMul_Serial_Wrapper(VecDotF32_FMA, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulF32_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
        => QuantizedMatMul_Parallel_Wrapper(VecDotF32_FMA, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulF16_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
        => QuantizedMatMul_Serial_Wrapper(VecDotF16_FMA, input, rawWeights, output, M, K, N);

    public static unsafe void QuantizedMatMulF16_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
        => QuantizedMatMul_Parallel_Wrapper(VecDotF16_FMA, input, rawWeights, output, M, K, N);

    public static void ReadF32_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        Span<byte> byteView = MemoryMarshal.AsBytes(data);
        reader.Read(byteView);
    }

    public static void ReadF16_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = HalfToFloat_Scalar(reader.ReadUInt16());
    }
    public static void ReadF16_F16C(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = HalfToFloat_F16C(reader.ReadUInt16());
    }
}
