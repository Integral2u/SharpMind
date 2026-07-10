using System.Runtime.CompilerServices;
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
        float* w = (float*)rawWeights;
        if (M <= 1)
        {
            float* pIn = input;
            float* pOut = output;
            for (int col = 0; col < N; col++)
            {
                float sum = 0;
                float* pW = w + (long)col * K;
                for (int i = 0; i < K; i++)
                    sum += pIn[i] * pW[i];
                pOut[col] = sum;
            }
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
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
        ushort* w = (ushort*)rawWeights;
        if (M <= 1)
        {
            float* pIn = input;
            float* pOut = output;
            for (int col = 0; col < N; col++)
            {
                float sum = 0;
                ushort* pW = w + (long)col * K;
                for (int i = 0; i < K; i++)
                    sum += pIn[i] * HalfToFloat_Scalar(pW[i]);
                pOut[col] = sum;
            }
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
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

    public static unsafe void ReadF32_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        Span<byte> byteView = MemoryMarshal.AsBytes(data);
        reader.Read(byteView);
    }

    public static unsafe void ReadF16_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = HalfToFloat_Scalar(reader.ReadUInt16());
    }
    public static unsafe void ReadF16_F16C(BinaryReader reader, Span<float> data, int n)
    {
        for (int i = 0; i < n; i++) data[i] = HalfToFloat_F16C(reader.ReadUInt16());
    }
}
