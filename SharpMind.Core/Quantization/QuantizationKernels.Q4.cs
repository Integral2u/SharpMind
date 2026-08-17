using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    private static readonly float[] kvalues_iq4nl =
        { -127f, -104f, -83f, -65f, -49f, -35f, -22f, -10f, 1f, 13f, 25f, 38f, 53f, 69f, 89f, 113f };


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_0_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                int half = QK / 2;
                bool lo = i < half;
                int qsOff0 = lo ? i : i - half;
                int shift0 = lo ? 0 : 4;
                bool lo8 = (i + 8) < half;
                int qsOff1 = lo8 ? i + 8 : i + 8 - half;
                int shift1 = lo8 ? 0 : 4;
                var w0 = Avx.Multiply(Vector256.Create(
                    (float)(((qs[qsOff0 + 0] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 1] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 2] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 3] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 4] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 5] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 6] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 7] >> shift0) & 0x0F) - 8)
                ), vd);
                var w1 = Avx.Multiply(Vector256.Create(
                    (float)(((qs[qsOff1 + 0] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 1] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 2] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 3] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 4] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 5] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 6] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 7] >> shift1) & 0x0F) - 8)
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w0));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 8]), w1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                int half = QK / 2;
                bool lo = i < half;
                int qsOff = lo ? i : i - half;
                int shift = lo ? 0 : 4;
                var w = Avx.Multiply(Vector256.Create(
                    (float)(((qs[qsOff + 0] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 1] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 2] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 3] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 4] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 5] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 6] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 7] >> shift) & 0x0F) - 8)
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_0_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                int half = QK / 2;
                bool lo = i < half;
                int qsOff0 = lo ? i : i - half;
                int shift0 = lo ? 0 : 4;
                bool lo8 = (i + 8) < half;
                int qsOff1 = lo8 ? i + 8 : i + 8 - half;
                int shift1 = lo8 ? 0 : 4;
                var w0 = Avx.Multiply(Vector256.Create(
                    (float)(((qs[qsOff0 + 0] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 1] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 2] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 3] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 4] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 5] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 6] >> shift0) & 0x0F) - 8),
                    (float)(((qs[qsOff0 + 7] >> shift0) & 0x0F) - 8)
                ), vd);
                var w1 = Avx.Multiply(Vector256.Create(
                    (float)(((qs[qsOff1 + 0] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 1] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 2] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 3] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 4] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 5] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 6] >> shift1) & 0x0F) - 8),
                    (float)(((qs[qsOff1 + 7] >> shift1) & 0x0F) - 8)
                ), vd);
                vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), w0, vacc0);
                vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i + 8]), w1, vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                int half = QK / 2;
                bool lo = i < half;
                int qsOff = lo ? i : i - half;
                int shift = lo ? 0 : 4;
                var w = Avx.Multiply(Vector256.Create(
                    (float)(((qs[qsOff + 0] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 1] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 2] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 3] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 4] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 5] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 6] >> shift) & 0x0F) - 8),
                    (float)(((qs[qsOff + 7] >> shift) & 0x0F) - 8)
                ), vd);
                vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), w, vacc0);
            }
            for (; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * ((q - 8) * d);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_0_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            int i = 0;
            for (; i <= blockEnd - 4; i += 4)
            {
                int half = QK / 2;
                bool lo = i < half;
                int qsOff = lo ? i : i - half;
                int shift = lo ? 0 : 4;
                float v0 = ((qs[qsOff + 0] >> shift) & 0x0F) - 8;
                float v1 = ((qs[qsOff + 1] >> shift) & 0x0F) - 8;
                float v2 = ((qs[qsOff + 2] >> shift) & 0x0F) - 8;
                float v3 = ((qs[qsOff + 3] >> shift) & 0x0F) - 8;
                var vv = Vector128.Create(v0, v1, v2, v3) * Vector128.Create(d);
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum128_Sse(Sse.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_1_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            float m = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* qs = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_1_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            float m = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* qs = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            var vm = Vector256.Create(m);

            int i = 0;
            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            for (; i <= blockEnd - 16; i += 16)
            {
                int half = QK / 2;
                bool lo0 = i < half;
                int qsOff0 = lo0 ? i : i - half;
                int shift0 = lo0 ? 0 : 4;
                bool lo8 = (i + 8) < half;
                int qsOff1 = lo8 ? i + 8 : i + 8 - half;
                int shift1 = lo8 ? 0 : 4;
                var w0 = Avx.Add(Avx.Multiply(Vector256.Create(
                    (float)((qs[qsOff0 + 0] >> shift0) & 0x0F), (float)((qs[qsOff0 + 1] >> shift0) & 0x0F),
                    (float)((qs[qsOff0 + 2] >> shift0) & 0x0F), (float)((qs[qsOff0 + 3] >> shift0) & 0x0F),
                    (float)((qs[qsOff0 + 4] >> shift0) & 0x0F), (float)((qs[qsOff0 + 5] >> shift0) & 0x0F),
                    (float)((qs[qsOff0 + 6] >> shift0) & 0x0F), (float)((qs[qsOff0 + 7] >> shift0) & 0x0F)
                ), vd), vm);
                var w1 = Avx.Add(Avx.Multiply(Vector256.Create(
                    (float)((qs[qsOff1 + 0] >> shift1) & 0x0F), (float)((qs[qsOff1 + 1] >> shift1) & 0x0F),
                    (float)((qs[qsOff1 + 2] >> shift1) & 0x0F), (float)((qs[qsOff1 + 3] >> shift1) & 0x0F),
                    (float)((qs[qsOff1 + 4] >> shift1) & 0x0F), (float)((qs[qsOff1 + 5] >> shift1) & 0x0F),
                    (float)((qs[qsOff1 + 6] >> shift1) & 0x0F), (float)((qs[qsOff1 + 7] >> shift1) & 0x0F)
                ), vd), vm);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w0));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 8]), w1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                int half = QK / 2;
                bool lo = i < half;
                int qsOff = lo ? i : i - half;
                int shift = lo ? 0 : 4;
                var w = Avx.Add(Avx.Multiply(Vector256.Create(
                    (float)((qs[qsOff + 0] >> shift) & 0x0F), (float)((qs[qsOff + 1] >> shift) & 0x0F),
                    (float)((qs[qsOff + 2] >> shift) & 0x0F), (float)((qs[qsOff + 3] >> shift) & 0x0F),
                    (float)((qs[qsOff + 4] >> shift) & 0x0F), (float)((qs[qsOff + 5] >> shift) & 0x0F),
                    (float)((qs[qsOff + 6] >> shift) & 0x0F), (float)((qs[qsOff + 7] >> shift) & 0x0F)
                ), vd), vm);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_1_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            float m = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* qs = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            var vm = Vector256.Create(m);

            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                int half = QK / 2;
                bool lo0 = i < half;
                int qsOff0 = lo0 ? i : i - half;
                int shift0 = lo0 ? 0 : 4;
                bool lo8 = (i + 8) < half;
                int qsOff1 = lo8 ? i + 8 : i + 8 - half;
                int shift1 = lo8 ? 0 : 4;
                var w0 = Avx.Add(Avx.Multiply(Vector256.Create(
                    (float)((qs[qsOff0 + 0] >> shift0) & 0x0F), (float)((qs[qsOff0 + 1] >> shift0) & 0x0F),
                    (float)((qs[qsOff0 + 2] >> shift0) & 0x0F), (float)((qs[qsOff0 + 3] >> shift0) & 0x0F),
                    (float)((qs[qsOff0 + 4] >> shift0) & 0x0F), (float)((qs[qsOff0 + 5] >> shift0) & 0x0F),
                    (float)((qs[qsOff0 + 6] >> shift0) & 0x0F), (float)((qs[qsOff0 + 7] >> shift0) & 0x0F)
                ), vd), vm);
                var w1 = Avx.Add(Avx.Multiply(Vector256.Create(
                    (float)((qs[qsOff1 + 0] >> shift1) & 0x0F), (float)((qs[qsOff1 + 1] >> shift1) & 0x0F),
                    (float)((qs[qsOff1 + 2] >> shift1) & 0x0F), (float)((qs[qsOff1 + 3] >> shift1) & 0x0F),
                    (float)((qs[qsOff1 + 4] >> shift1) & 0x0F), (float)((qs[qsOff1 + 5] >> shift1) & 0x0F),
                    (float)((qs[qsOff1 + 6] >> shift1) & 0x0F), (float)((qs[qsOff1 + 7] >> shift1) & 0x0F)
                ), vd), vm);
                vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), w0, vacc0);
                vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i + 8]), w1, vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                int half = QK / 2;
                bool lo = i < half;
                int qsOff = lo ? i : i - half;
                int shift = lo ? 0 : 4;
                var w = Avx.Add(Avx.Multiply(Vector256.Create(
                    (float)((qs[qsOff + 0] >> shift) & 0x0F), (float)((qs[qsOff + 1] >> shift) & 0x0F),
                    (float)((qs[qsOff + 2] >> shift) & 0x0F), (float)((qs[qsOff + 3] >> shift) & 0x0F),
                    (float)((qs[qsOff + 4] >> shift) & 0x0F), (float)((qs[qsOff + 5] >> shift) & 0x0F),
                    (float)((qs[qsOff + 6] >> shift) & 0x0F), (float)((qs[qsOff + 7] >> shift) & 0x0F)
                ), vd), vm);
                vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), w, vacc0);
            }
            for (; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * (q * d + m);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_1_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            float m = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* qs = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector128.Create(d);
            var vm = Vector128.Create(m);

            int i = 0;
            for (; i <= blockEnd - 4; i += 4)
            {
                int half = QK / 2;
                bool lo = i < half;
                int qsOff = lo ? i : i - half;
                int shift = lo ? 0 : 4;
                float v0 = (qs[qsOff + 0] >> shift) & 0x0F;
                float v1 = (qs[qsOff + 1] >> shift) & 0x0F;
                float v2 = (qs[qsOff + 2] >> shift) & 0x0F;
                float v3 = (qs[qsOff + 3] >> shift) & 0x0F;
                var vv = Sse.Add(Sse.Multiply(Vector128.Create(v0, v1, v2, v3), vd), vm);
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum128_Sse(Sse.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_NL_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * (d * kvalues_iq4nl[nib]);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_NL_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            var vacc0 = Vector256<float>.Zero;
            var vacc1 = Vector256<float>.Zero;
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                int half = QK / 2;
                var w0 = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)],
                    kvalues_iq4nl[(i+1 < half) ? (qs[i+1] & 0x0F) : (qs[i+1 - half] >> 4)],
                    kvalues_iq4nl[(i+2 < half) ? (qs[i+2] & 0x0F) : (qs[i+2 - half] >> 4)],
                    kvalues_iq4nl[(i+3 < half) ? (qs[i+3] & 0x0F) : (qs[i+3 - half] >> 4)],
                    kvalues_iq4nl[(i+4 < half) ? (qs[i+4] & 0x0F) : (qs[i+4 - half] >> 4)],
                    kvalues_iq4nl[(i+5 < half) ? (qs[i+5] & 0x0F) : (qs[i+5 - half] >> 4)],
                    kvalues_iq4nl[(i+6 < half) ? (qs[i+6] & 0x0F) : (qs[i+6 - half] >> 4)],
                    kvalues_iq4nl[(i+7 < half) ? (qs[i+7] & 0x0F) : (qs[i+7 - half] >> 4)]
                ), vd);
                var w1 = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(i+8 < half) ? (qs[i+8] & 0x0F) : (qs[i+8 - half] >> 4)],
                    kvalues_iq4nl[(i+9 < half) ? (qs[i+9] & 0x0F) : (qs[i+9 - half] >> 4)],
                    kvalues_iq4nl[(i+10 < half) ? (qs[i+10] & 0x0F) : (qs[i+10 - half] >> 4)],
                    kvalues_iq4nl[(i+11 < half) ? (qs[i+11] & 0x0F) : (qs[i+11 - half] >> 4)],
                    kvalues_iq4nl[(i+12 < half) ? (qs[i+12] & 0x0F) : (qs[i+12 - half] >> 4)],
                    kvalues_iq4nl[(i+13 < half) ? (qs[i+13] & 0x0F) : (qs[i+13 - half] >> 4)],
                    kvalues_iq4nl[(i+14 < half) ? (qs[i+14] & 0x0F) : (qs[i+14 - half] >> 4)],
                    kvalues_iq4nl[(i+15 < half) ? (qs[i+15] & 0x0F) : (qs[i+15 - half] >> 4)]
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w0));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 8]), w1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                int half = QK / 2;
                var w = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)],
                    kvalues_iq4nl[(i+1 < half) ? (qs[i+1] & 0x0F) : (qs[i+1 - half] >> 4)],
                    kvalues_iq4nl[(i+2 < half) ? (qs[i+2] & 0x0F) : (qs[i+2 - half] >> 4)],
                    kvalues_iq4nl[(i+3 < half) ? (qs[i+3] & 0x0F) : (qs[i+3 - half] >> 4)],
                    kvalues_iq4nl[(i+4 < half) ? (qs[i+4] & 0x0F) : (qs[i+4 - half] >> 4)],
                    kvalues_iq4nl[(i+5 < half) ? (qs[i+5] & 0x0F) : (qs[i+5 - half] >> 4)],
                    kvalues_iq4nl[(i+6 < half) ? (qs[i+6] & 0x0F) : (qs[i+6 - half] >> 4)],
                    kvalues_iq4nl[(i+7 < half) ? (qs[i+7] & 0x0F) : (qs[i+7 - half] >> 4)]
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * (d * kvalues_iq4nl[nib]);
            }
        }
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4_NL_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_F16C(*(ushort*)block);
            byte* qs = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;
            var vd = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 16; i += 16)
            {
                int half = QK / 2;
                var w0 = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)],
                    kvalues_iq4nl[(i+1 < half) ? (qs[i+1] & 0x0F) : (qs[i+1 - half] >> 4)],
                    kvalues_iq4nl[(i+2 < half) ? (qs[i+2] & 0x0F) : (qs[i+2 - half] >> 4)],
                    kvalues_iq4nl[(i+3 < half) ? (qs[i+3] & 0x0F) : (qs[i+3 - half] >> 4)],
                    kvalues_iq4nl[(i+4 < half) ? (qs[i+4] & 0x0F) : (qs[i+4 - half] >> 4)],
                    kvalues_iq4nl[(i+5 < half) ? (qs[i+5] & 0x0F) : (qs[i+5 - half] >> 4)],
                    kvalues_iq4nl[(i+6 < half) ? (qs[i+6] & 0x0F) : (qs[i+6 - half] >> 4)],
                    kvalues_iq4nl[(i+7 < half) ? (qs[i+7] & 0x0F) : (qs[i+7 - half] >> 4)]
                ), vd);
                var w1 = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(i+8 < half) ? (qs[i+8] & 0x0F) : (qs[i+8 - half] >> 4)],
                    kvalues_iq4nl[(i+9 < half) ? (qs[i+9] & 0x0F) : (qs[i+9 - half] >> 4)],
                    kvalues_iq4nl[(i+10 < half) ? (qs[i+10] & 0x0F) : (qs[i+10 - half] >> 4)],
                    kvalues_iq4nl[(i+11 < half) ? (qs[i+11] & 0x0F) : (qs[i+11 - half] >> 4)],
                    kvalues_iq4nl[(i+12 < half) ? (qs[i+12] & 0x0F) : (qs[i+12 - half] >> 4)],
                    kvalues_iq4nl[(i+13 < half) ? (qs[i+13] & 0x0F) : (qs[i+13 - half] >> 4)],
                    kvalues_iq4nl[(i+14 < half) ? (qs[i+14] & 0x0F) : (qs[i+14 - half] >> 4)],
                    kvalues_iq4nl[(i+15 < half) ? (qs[i+15] & 0x0F) : (qs[i+15 - half] >> 4)]
                ), vd);
                vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), w0, vacc0);
                vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i + 8]), w1, vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                int half = QK / 2;
                var w = Avx.Multiply(Vector256.Create(
                    kvalues_iq4nl[(i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)],
                    kvalues_iq4nl[(i+1 < half) ? (qs[i+1] & 0x0F) : (qs[i+1 - half] >> 4)],
                    kvalues_iq4nl[(i+2 < half) ? (qs[i+2] & 0x0F) : (qs[i+2 - half] >> 4)],
                    kvalues_iq4nl[(i+3 < half) ? (qs[i+3] & 0x0F) : (qs[i+3 - half] >> 4)],
                    kvalues_iq4nl[(i+4 < half) ? (qs[i+4] & 0x0F) : (qs[i+4 - half] >> 4)],
                    kvalues_iq4nl[(i+5 < half) ? (qs[i+5] & 0x0F) : (qs[i+5 - half] >> 4)],
                    kvalues_iq4nl[(i+6 < half) ? (qs[i+6] & 0x0F) : (qs[i+6 - half] >> 4)],
                    kvalues_iq4nl[(i+7 < half) ? (qs[i+7] & 0x0F) : (qs[i+7 - half] >> 4)]
                ), vd);
                vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[i]), w, vacc0);
            }
            for (; i < blockEnd; i++)
            {
                int nib = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += pIn[i] * (d * kvalues_iq4nl[nib]);
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float dSuper = HalfToFloat_Scalar(*(ushort*)(block + 0));
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 16;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s = GetScaleMinK4_Scale_Scalar(isc, scales);
                    float m = GetScaleMinK4_Min_Scalar(isc, scales);
                    for (int l = 0; l < 32 && basePos + l < blockEnd; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 64) * 32 + (idx % 32);
                        int qsShift = ((idx % 64) / 32) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += input[b * QK_K + idx - colBlockStart] * (s * v * dSuper - m * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    /// <summary>
    /// Eight consecutive Q4_K nibbles as floats. A 32-value sub-block that starts
    /// on a 32 boundary lives in one contiguous run of 32 bytes: chunk
    /// <c>basePos/64</c> of <paramref name="qs"/>, low nibbles for the first 32
    /// values of the chunk and high nibbles for the next 32. So group
    /// <paramref name="g"/> of the sub-block is bytes <c>[8g, 8g+8)</c> of that
    /// run, widened straight from memory (vpmovzxbd) and masked or shifted —
    /// never decoded one weight at a time into a stack buffer, which cannot
    /// store-forward into the vector load that follows it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> Q4KNibbles8(byte* qs, int basePos, int g)
    {
        var v = Avx2.ConvertToVector256Int32(qs + (basePos >> 6) * 32 + g * 8);
        v = (basePos & 32) == 0
            ? Avx2.And(v, Vector256.Create(0x0F))
            : Avx2.ShiftRightLogical(v, 4);   // bytes are < 256, so this is the high nibble
        return Avx.ConvertToVector256Single(v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 0));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 16;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s = GetScaleMinK4_Scale_Scalar(isc, scales);
                    float m = GetScaleMinK4_Min_Scalar(isc, scales);
                    var vs = Vector256.Create(s * dSuper);
                    var vm = Vector256.Create(m * minSuper);

                    int subRem = Math.Min(32, blockEnd - basePos);
                    int l = 0;
                    if ((basePos & 31) == 0)
                    {
                        for (; l <= subRem - 16; l += 16)
                        {
                            var vw0 = Avx.Subtract(Avx.Multiply(Q4KNibbles8(qs, basePos, l >> 3), vs), vm);
                            var vw1 = Avx.Subtract(Avx.Multiply(Q4KNibbles8(qs, basePos, (l >> 3) + 1), vs), vm);
                            vacc0 = Avx.Add(Avx.Multiply(Vector256.LoadUnsafe(ref pIn[basePos + l]), vw0), vacc0);
                            vacc1 = Avx.Add(Avx.Multiply(Vector256.LoadUnsafe(ref pIn[basePos + l + 8]), vw1), vacc1);
                        }
                        for (; l <= subRem - 8; l += 8)
                        {
                            var vw = Avx.Subtract(Avx.Multiply(Q4KNibbles8(qs, basePos, l >> 3), vs), vm);
                            vacc0 = Avx.Add(Avx.Multiply(Vector256.LoadUnsafe(ref pIn[basePos + l]), vw), vacc0);
                        }
                    }
                    for (; l < subRem; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 64) * 32 + (idx % 32);
                        int qsShift = ((idx % 64) / 32) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += pIn[idx] * (s * v * dSuper - m * minSuper);
                    }
                }
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ4K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        var vacc0 = Vector256<float>.Zero;
        var vacc1 = Vector256<float>.Zero;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 0));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 2));
            byte* scales = block + 4;
            byte* qs = block + 16;

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s = GetScaleMinK4_Scale_Scalar(isc, scales);
                    float m = GetScaleMinK4_Min_Scalar(isc, scales);
                    var vs = Vector256.Create(s * dSuper);
                    var vm = Vector256.Create(m * minSuper);

                    int subRem = Math.Min(32, blockEnd - basePos);
                    int l = 0;
                    if ((basePos & 31) == 0)
                    {
                        for (; l <= subRem - 16; l += 16)
                        {
                            var vw0 = Fma.MultiplySubtract(Q4KNibbles8(qs, basePos, l >> 3), vs, vm);
                            var vw1 = Fma.MultiplySubtract(Q4KNibbles8(qs, basePos, (l >> 3) + 1), vs, vm);
                            vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[basePos + l]), vw0, vacc0);
                            vacc1 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[basePos + l + 8]), vw1, vacc1);
                        }
                        for (; l <= subRem - 8; l += 8)
                        {
                            var vw = Fma.MultiplySubtract(Q4KNibbles8(qs, basePos, l >> 3), vs, vm);
                            vacc0 = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref pIn[basePos + l]), vw, vacc0);
                        }
                    }
                    for (; l < subRem; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 64) * 32 + (idx % 32);
                        int qsShift = ((idx % 64) / 32) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += pIn[idx] * (s * v * dSuper - m * minSuper);
                    }
                }
            }
        }
        sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
        return (float)sum;
    }

    public static unsafe void QuantizedMatMulQ4_0_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_0_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_0_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_1_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_1_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_1_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_NL_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_NL_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_NL_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Serial_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4K_Scalar(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Parallel_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4K_Scalar, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4K_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }

    public static void ReadQ4_0_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 18;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int half = qk / 2;
                int nib = (j < half) ? (buf[2 + j] & 0x0F) : (buf[2 + j - half] >> 4);
                data[blockStart + j] = (nib - 8) * d;
            }
        }
    }

    public static void ReadQ4_1_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 20;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            float m = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[2]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int half = qk / 2;
                int q = (j < half) ? (buf[4 + j] & 0x0F) : (buf[4 + j - half] >> 4);
                data[blockStart + j] = q * d + m;
            }
        }
    }

    public static void ReadQ4_NL_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int qk = 32;
        const int blockBytes = 18;
        int nBlocks = (n + qk - 1) / qk;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * qk;
            reader.Read(buf);
            float d = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            int valid = Math.Min(qk, n - blockStart);

            for (int j = 0; j < valid; j++)
            {
                int half = qk / 2;
                int nib = (j < half) ? (buf[2 + j] & 0x0F) : (buf[2 + j - half] >> 4);
                data[blockStart + j] = d * kvalues_iq4nl[nib];
            }
        }
    }

    public static unsafe void ReadQ4K_Scalar(BinaryReader reader, Span<float> data, int n)
    {
        const int QK_K = 256;
        const int blockBytes = 144;
        int nBlocks = (n + QK_K - 1) / QK_K;
        Span<byte> buf = stackalloc byte[blockBytes];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * QK_K;
            reader.Read(buf);
            float dSuper = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[0]));
            float minSuper = HalfToFloat_Scalar(Unsafe.ReadUnaligned<ushort>(ref buf[2]));

            fixed (byte* pBuf = buf)
            {
                byte* scales = pBuf + 4;
                byte* qs = pBuf + 16;
                for (int i = 0; i < QK_K && blockStart + i < n; i++)
                {
                    int sub = i / 32;
                    float s = GetScaleMinK4_Scale_Scalar(sub, scales);
                    float m = GetScaleMinK4_Min_Scalar(sub, scales);
                    int qsByte = (i / 64) * 32 + (i % 32);
                    int qsShift = ((i % 64) / 32) * 4;
                    int v = (qs[qsByte] >> qsShift) & 0x0F;
                    data[blockStart + i] = s * v * dSuper - m * minSuper;
                }
            }
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_0_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_0_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_0_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Serial_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_0_SSE(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Parallel_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_0_SSE, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_0_SSE(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_1_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_1_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_1_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Serial_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_1_SSE(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Parallel_SSE(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_1_SSE, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_1_SSE(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_NL_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_NL_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_NL_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Serial_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4K_AVX2(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Parallel_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4K_AVX2, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4K_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4K_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4K_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4K_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4K_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_0_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_0_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_0_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_0_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_1_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_1_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_1_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_1_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Serial_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ4_NL_FMA(pInRow, rawWeights, col, K);
        }
    }

    public static unsafe void QuantizedMatMulQ4_NL_Parallel_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ4_NL_FMA, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ4_NL_FMA(pInRow, rawWeights, col, K);
            });
        }
    }
}
