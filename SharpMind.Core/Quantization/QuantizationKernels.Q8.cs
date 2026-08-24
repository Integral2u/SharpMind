using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
                sum += input[b * QK + i] * (values[i] * d);
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_0_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i + 8));
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, Avx.Multiply(vw0, vd)));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, Avx.Multiply(vw1, vd)));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, Avx.Multiply(vw, vd)));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_0_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i + 8));
                vacc0 = Fma.MultiplyAdd(vi0, Avx.Multiply(vw0, vd), vacc0);
                vacc1 = Fma.MultiplyAdd(vi1, Avx.Multiply(vw1, vd), vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                vacc0 = Fma.MultiplyAdd(vi, Avx.Multiply(vw, vd), vacc0);
            }
            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_0_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vacc = Vector128<float>.Zero;
            var vd = Vector128.Create(d);
            int i = 0;
            for (; i <= blockEnd - 4; i += 4)
            {
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                var vw = Vector128.Create(
                    (float)values[i], (float)values[i + 1], (float)values[i + 2], (float)values[i + 3]);
                var vs = Sse.Multiply(vw, vd);
                vacc = Sse.Add(vacc, Sse.Multiply(vi, vs));
            }
            sum += MathHelpers.HSum128_Sse(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_1_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* qs = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
                sum += input[b * QK + i] * (qs[i] * d);
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_1_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            sbyte* qs = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i + 8));
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, Avx.Multiply(vw0, vd)));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, Avx.Multiply(vw1, vd)));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, Avx.Multiply(vw, vd)));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_1_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            sbyte* qs = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i + 8));
                vacc0 = Fma.MultiplyAdd(vi0, Avx.Multiply(vw0, vd), vacc0);
                vacc1 = Fma.MultiplyAdd(vi1, Avx.Multiply(vw1, vd), vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                vacc0 = Fma.MultiplyAdd(vi, Avx.Multiply(vw, vd), vacc0);
            }
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_1_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* qs = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vacc = Vector128<float>.Zero;
            var vd = Vector128.Create(d);
            int i = 0;
            for (; i <= blockEnd - 4; i += 4)
            {
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                var vw = Vector128.Create(
                    (float)qs[i], (float)qs[i + 1], (float)qs[i + 2], (float)qs[i + 3]);
                var vs = Sse.Multiply(vw, vd);
                vacc = Sse.Add(vacc, Sse.Multiply(vi, vs));
            }
            sum += MathHelpers.HSum128_Sse(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float d = *(float*)block;
            sbyte* qs = (sbyte*)(block + 4);
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
                sum += input[b * QK_K + i - colBlockStart] * (qs[i] * d);
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float d = *(float*)block;
            sbyte* qs = (sbyte*)(block + 4);
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            var vd = Vector256.Create(d);
            int i = curBlockStart;
            for (; i <= blockEnd - 16; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i + 8));
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, Avx.Multiply(vw0, vd)));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, Avx.Multiply(vw1, vd)));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, Avx.Multiply(vw, vd)));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float d = *(float*)block;
            sbyte* qs = (sbyte*)(block + 4);
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            var vd = Vector256.Create(d);
            int i = curBlockStart;
            for (; i <= blockEnd - 16; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i + 8));
                vacc0 = Fma.MultiplyAdd(vi0, Avx.Multiply(vw0, vd), vacc0);
                vacc1 = Fma.MultiplyAdd(vi1, Avx.Multiply(vw1, vd), vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                vacc0 = Fma.MultiplyAdd(vi, Avx.Multiply(vw, vd), vacc0);
            }
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8K_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float d = *(float*)block;
            sbyte* qs = (sbyte*)(block + 4);
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;

            var vacc = Vector128<float>.Zero;
            var vd = Vector128.Create(d);
            int i = curBlockStart;
            for (; i <= blockEnd - 4; i += 4)
            {
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                var vw = Vector128.Create(
                    (float)qs[i], (float)qs[i + 1], (float)qs[i + 2], (float)qs[i + 3]);
                var vs = Sse.Multiply(vw, vd);
                vacc = Sse.Add(vacc, Sse.Multiply(vi, vs));
            }
            sum += MathHelpers.HSum128_Sse(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

    public static unsafe void ReadQ8_0_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 34;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            fixed (byte* pBuf = buf)
            {
                sbyte* values = (sbyte*)(pBuf + 2);
                for (int j = 0; j < valid; j++)
                    data[blockStart + j] = values[j] * d;
            }
        }
    }

    public static void ReadQ8_1_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 36;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
                data[blockStart + j] = (sbyte)buf[4 + j] * d;
        }
    }

    public static unsafe void ReadQ8K_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 256;
        const int blockBytes = 292;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            int valid = Math.Min(qk, n - blockStart);

            fixed (byte* pBuf = buf)
            {
                float d = *(float*)pBuf;
                sbyte* values = (sbyte*)(pBuf + 4);
                for (int j = 0; j < valid; j++)
                    data[blockStart + j] = values[j] * d;
            }
        }
    }

    public static unsafe void QuantizedMatMulQ8_0_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_0_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8_0_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_0_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8_0_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_0_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8_0_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_0_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_0_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8_0_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            QuantizedMatMul_Serial_Wrapper(VecDotQ8_0_FMA, input, rawWeights, output, M, K, N);
            return;
        }
        // M > 1 (prefill/training): dequantize each column once, then run the
        // blocked four-row microkernel. See QuantBlockedColumns.
        QuantBlockedMatMul(DequantColumnQ8_0, input, rawWeights, output, M, K, N, parallel: false);
    }

    public static unsafe void QuantizedMatMulQ8_0_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_0_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            QuantBlockedMatMul(DequantColumnQ8_0, input, rawWeights, output, M, K, N, parallel: true);
        }
    }

    public static unsafe void QuantizedMatMulQ8_1_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_1_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8_1_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_1_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_1_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8K_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8K_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8K_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8K_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8K_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8_1_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_1_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8_1_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_1_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_1_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8_1_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_1_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8_1_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_1_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_1_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8_1_Serial_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_1_SSE(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8_1_Parallel_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_1_SSE, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_1_SSE(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8K_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8K_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8K_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8K_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8K_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8K_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8K_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8K_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8K_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8K_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8K_Serial_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8K_SSE(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ8K_Parallel_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8K_SSE, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8K_SSE(pInRow, rawWeights, col, K);
            });
        }
    }
}
