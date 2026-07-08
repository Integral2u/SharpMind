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
}
