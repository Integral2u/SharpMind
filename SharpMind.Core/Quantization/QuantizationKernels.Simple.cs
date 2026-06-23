using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{

    // QuantizedMatMulQ8_0 — fused matmul for q8_0 weights
    // Processes all output columns in one call, parallelized over column
    // groups (NR=8). Eliminates per-column VecDot function call overhead
    // and enables multi-core scaling.

    public static unsafe void QuantizedMatMulQ8_0_FMA(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        const int QK = 32;
        const int BLOCK_BYTES = 34;
        const int NR = 8;
        int nBlocks = (K + QK - 1) / QK;
        int colStride = nBlocks * BLOCK_BYTES;

        if (M <= 1)
        {
            int nGroups = N / NR;
            if (nGroups > 0)
            {
                System.Threading.Tasks.Parallel.For(0, nGroups, group =>
                {
                    int colBase = group * NR;
                    double* sums = stackalloc double[NR];
                    for (int ci = 0; ci < NR; ci++) sums[ci] = 0.0;

                    for (int b = 0; b < nBlocks; b++)
                    {
                        float* pInBlock = input + b * QK;
                        bool isLast = b == nBlocks - 1;
                        int blockEnd = isLast ? K - b * QK : QK;
                        bool fullBlock = blockEnd == QK;

                        Vector256<float> vi0 = default, vi1 = default, vi2 = default, vi3 = default;
                        if (fullBlock)
                        {
                            vi0 = Vector256.LoadUnsafe(ref pInBlock[0]);
                            vi1 = Vector256.LoadUnsafe(ref pInBlock[8]);
                            vi2 = Vector256.LoadUnsafe(ref pInBlock[16]);
                            vi3 = Vector256.LoadUnsafe(ref pInBlock[24]);
                        }

                        for (int ci = 0; ci < NR; ci++)
                        {
                            int col = colBase + ci;
                            byte* block = rawWeights + (long)col * colStride + b * BLOCK_BYTES;
                            float d = HalfToFloat_F16C(*(ushort*)block);
                            sbyte* values = (sbyte*)(block + 2);
                            var vd = Vector256.Create(d);

                            float s;
                            if (fullBlock)
                            {
                                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values));
                                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + 8));
                                var vw2 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + 16));
                                var vw3 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + 24));

                                var a0 = Fma.MultiplyAdd(vi0, Avx.Multiply(vw0, vd), Vector256<float>.Zero);
                                var a1 = Fma.MultiplyAdd(vi1, Avx.Multiply(vw1, vd), Vector256<float>.Zero);
                                var a2 = Fma.MultiplyAdd(vi2, Avx.Multiply(vw2, vd), Vector256<float>.Zero);
                                var a3 = Fma.MultiplyAdd(vi3, Avx.Multiply(vw3, vd), Vector256<float>.Zero);

                                s = MathHelpers.HSum256_Avx((a0 + a1) + (a2 + a3));
                            }
                            else
                            {
                                var vacc = Vector256<float>.Zero;
                                int i = 0;
                                for (; i <= blockEnd - 16; i += 8)
                                {
                                    var vi = Vector256.LoadUnsafe(ref pInBlock[i]);
                                    var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                                    vacc = Fma.MultiplyAdd(vi, Avx.Multiply(vw, vd), vacc);
                                }
                                s = MathHelpers.HSum256_Avx(vacc);
                                for (; i < blockEnd; i++) s += pInBlock[i] * (values[i] * d);
                            }

                            sums[ci] += s;
                        }
                    }

                    for (int ci = 0; ci < NR; ci++)
                        output[colBase + ci] = (float)sums[ci];
                });
            }

            int tailStart = nGroups * NR;
            for (int col = tailStart; col < N; col++)
                output[col] = VecDotQ8_0_FMA(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                double* sums = stackalloc double[NR];

                int nGroups = N / NR;
                for (int group = 0; group < nGroups; group++)
                {
                    int colBase = group * NR;
                    for (int ci = 0; ci < NR; ci++) sums[ci] = 0.0;

                    for (int b = 0; b < nBlocks; b++)
                    {
                        float* pInBlock = pInRow + b * QK;
                        bool isLast = b == nBlocks - 1;
                        int blockEnd = isLast ? K - b * QK : QK;
                        bool fullBlock = blockEnd == QK;

                        Vector256<float> vi0 = default, vi1 = default, vi2 = default, vi3 = default;
                        if (fullBlock)
                        {
                            vi0 = Vector256.LoadUnsafe(ref pInBlock[0]);
                            vi1 = Vector256.LoadUnsafe(ref pInBlock[8]);
                            vi2 = Vector256.LoadUnsafe(ref pInBlock[16]);
                            vi3 = Vector256.LoadUnsafe(ref pInBlock[24]);
                        }

                        for (int ci = 0; ci < NR; ci++)
                        {
                            int col = colBase + ci;
                            byte* block = rawWeights + (long)col * colStride + b * BLOCK_BYTES;
                            float d = HalfToFloat_F16C(*(ushort*)block);
                            sbyte* values = (sbyte*)(block + 2);
                            var vd = Vector256.Create(d);

                            float s;
                            if (fullBlock)
                            {
                                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values));
                                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + 8));
                                var vw2 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + 16));
                                var vw3 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + 24));

                                var a0 = Fma.MultiplyAdd(vi0, Avx.Multiply(vw0, vd), Vector256<float>.Zero);
                                var a1 = Fma.MultiplyAdd(vi1, Avx.Multiply(vw1, vd), Vector256<float>.Zero);
                                var a2 = Fma.MultiplyAdd(vi2, Avx.Multiply(vw2, vd), Vector256<float>.Zero);
                                var a3 = Fma.MultiplyAdd(vi3, Avx.Multiply(vw3, vd), Vector256<float>.Zero);

                                s = MathHelpers.HSum256_Avx((a0 + a1) + (a2 + a3));
                            }
                            else
                            {
                                var vacc = Vector256<float>.Zero;
                                int i = 0;
                                for (; i <= blockEnd - 16; i += 8)
                                {
                                    var vi = Vector256.LoadUnsafe(ref pInBlock[i]);
                                    var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                                    vacc = Fma.MultiplyAdd(vi, Avx.Multiply(vw, vd), vacc);
                                }
                                s = MathHelpers.HSum256_Avx(vacc);
                                for (; i < blockEnd; i++) s += pInBlock[i] * (values[i] * d);
                            }

                            sums[ci] += s;
                        }
                    }

                    for (int ci = 0; ci < NR; ci++)
                        pOutRow[colBase + ci] = (float)sums[ci];
                }

                int tail = nGroups * NR;
                for (int col = tail; col < N; col++)
                    pOutRow[col] = VecDotQ8_0_FMA(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8_0_AVX2(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        const int QK = 32;
        const int BLOCK_BYTES = 34;
        const int NR = 8;
        int nBlocks = (K + QK - 1) / QK;
        int colStride = nBlocks * BLOCK_BYTES;

        if (M <= 1)
        {
            int nGroups = N / NR;
            if (nGroups > 0)
            {
                System.Threading.Tasks.Parallel.For(0, nGroups, group =>
                {
                    int colBase = group * NR;
                    double* sums = stackalloc double[NR];
                    for (int ci = 0; ci < NR; ci++) sums[ci] = 0.0;

                    for (int b = 0; b < nBlocks; b++)
                    {
                        float* pInBlock = input + b * QK;
                        int blockEnd = Math.Min(QK, K - b * QK);

                        for (int ci = 0; ci < NR; ci++)
                        {
                            int col = colBase + ci;
                            byte* block = rawWeights + (long)col * colStride + b * BLOCK_BYTES;
                            float d = HalfToFloat_F16C(*(ushort*)block);
                            sbyte* values = (sbyte*)(block + 2);

                            var vacc0 = Vector256<float>.Zero;
                            var vacc1 = Vector256<float>.Zero;
                            var vd = Vector256.Create(d);
                            int i = 0;
                            for (; i <= blockEnd - 24; i += 16)
                            {
                                var vi0 = Vector256.LoadUnsafe(ref pInBlock[i]);
                                var vi1 = Vector256.LoadUnsafe(ref pInBlock[i + 8]);
                                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i + 8));
                                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, Avx.Multiply(vw0, vd)));
                                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, Avx.Multiply(vw1, vd)));
                            }
                            for (; i <= blockEnd - 16; i += 8)
                            {
                                var vi = Vector256.LoadUnsafe(ref pInBlock[i]);
                                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, Avx.Multiply(vw, vd)));
                            }
                            float s = MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
                            for (; i < blockEnd; i++) s += pInBlock[i] * (values[i] * d);
                            sums[ci] += s;
                        }
                    }

                    for (int ci = 0; ci < NR; ci++)
                        output[colBase + ci] = (float)sums[ci];
                });
            }
            int tailStart = nGroups * NR;
            for (int col = tailStart; col < N; col++)
                output[col] = VecDotQ8_0_AVX2(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                double* sums = stackalloc double[NR];

                int nGroups = N / NR;
                for (int group = 0; group < nGroups; group++)
                {
                    int colBase = group * NR;
                    for (int ci = 0; ci < NR; ci++) sums[ci] = 0.0;

                    for (int b = 0; b < nBlocks; b++)
                    {
                        float* pInBlock = pInRow + b * QK;
                        int blockEnd = Math.Min(QK, K - b * QK);

                        for (int ci = 0; ci < NR; ci++)
                        {
                            int col = colBase + ci;
                            byte* block = rawWeights + (long)col * colStride + b * BLOCK_BYTES;
                            float d = HalfToFloat_F16C(*(ushort*)block);
                            sbyte* values = (sbyte*)(block + 2);

                            var vacc0 = Vector256<float>.Zero;
                            var vacc1 = Vector256<float>.Zero;
                            var vd = Vector256.Create(d);
                            int i = 0;
                            for (; i <= blockEnd - 24; i += 16)
                            {
                                var vi0 = Vector256.LoadUnsafe(ref pInBlock[i]);
                                var vi1 = Vector256.LoadUnsafe(ref pInBlock[i + 8]);
                                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i + 8));
                                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, Avx.Multiply(vw0, vd)));
                                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, Avx.Multiply(vw1, vd)));
                            }
                            for (; i <= blockEnd - 16; i += 8)
                            {
                                var vi = Vector256.LoadUnsafe(ref pInBlock[i]);
                                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi, Avx.Multiply(vw, vd)));
                            }
                            float s = MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
                            for (; i < blockEnd; i++) s += pInBlock[i] * (values[i] * d);
                            sums[ci] += s;
                        }
                    }

                    for (int ci = 0; ci < NR; ci++)
                        pOutRow[colBase + ci] = (float)sums[ci];
                }

                int tail = nGroups * NR;
                for (int col = tail; col < N; col++)
                    pOutRow[col] = VecDotQ8_0_AVX2(pInRow, rawWeights, col, K);
            });
        }
    }

    public static unsafe void QuantizedMatMulQ8_0_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        const int QK = 32;
        const int BLOCK_BYTES = 34;
        const int NR = 8;
        int nBlocks = (K + QK - 1) / QK;
        int colStride = nBlocks * BLOCK_BYTES;

        if (M <= 1)
        {
            int nGroups = N / NR;
            if (nGroups > 0)
            {
                System.Threading.Tasks.Parallel.For(0, nGroups, group =>
                {
                    int colBase = group * NR;
                    double* sums = stackalloc double[NR];
                    for (int ci = 0; ci < NR; ci++) sums[ci] = 0.0;

                    for (int b = 0; b < nBlocks; b++)
                    {
                        float* pInBlock = input + b * QK;
                        int blockEnd = Math.Min(QK, K - b * QK);

                        for (int ci = 0; ci < NR; ci++)
                        {
                            int col = colBase + ci;
                            byte* block = rawWeights + (long)col * colStride + b * BLOCK_BYTES;
                            float d = HalfToFloat_Scalar(*(ushort*)block);
                            sbyte* values = (sbyte*)(block + 2);

                            double s = 0.0;
                            for (int i = 0; i < blockEnd; i++)
                                s += pInBlock[i] * (values[i] * d);
                            sums[ci] += s;
                        }
                    }

                    for (int ci = 0; ci < NR; ci++)
                        output[colBase + ci] = (float)sums[ci];
                });
            }
            int tailStart = nGroups * NR;
            for (int col = tailStart; col < N; col++)
                output[col] = VecDotQ8_0_Scalar(input, rawWeights, col, K);
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                double* sums = stackalloc double[NR];

                int nGroups = N / NR;
                for (int group = 0; group < nGroups; group++)
                {
                    int colBase = group * NR;
                    for (int ci = 0; ci < NR; ci++) sums[ci] = 0.0;

                    for (int b = 0; b < nBlocks; b++)
                    {
                        float* pInBlock = pInRow + b * QK;
                        int blockEnd = Math.Min(QK, K - b * QK);

                        for (int ci = 0; ci < NR; ci++)
                        {
                            int col = colBase + ci;
                            byte* block = rawWeights + (long)col * colStride + b * BLOCK_BYTES;
                            float d = HalfToFloat_Scalar(*(ushort*)block);
                            sbyte* values = (sbyte*)(block + 2);

                            double s = 0.0;
                            for (int i = 0; i < blockEnd; i++)
                                s += pInBlock[i] * (values[i] * d);
                            sums[ci] += s;
                        }
                    }

                    for (int ci = 0; ci < NR; ci++)
                        pOutRow[colBase + ci] = (float)sums[ci];
                }

                int tail = nGroups * NR;
                for (int col = tail; col < N; col++)
                    pOutRow[col] = VecDotQ8_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }


    // VecDotQ8_0 — 8-bit block (QK=32)


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
            for (; i <= blockEnd - 24; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i + 8));
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, Avx.Multiply(vw0, vd)));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, Avx.Multiply(vw1, vd)));
            }
            for (; i <= blockEnd - 16; i += 8)
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

    public static unsafe float VecDotQ8_0_FMA(float* input, byte* rawWeights, int col, int inFeatures)
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
            for (; i <= blockEnd - 24; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i + 8));
                vacc0 = Fma.MultiplyAdd(vi0, Avx.Multiply(vw0, vd), vacc0);
                vacc1 = Fma.MultiplyAdd(vi1, Avx.Multiply(vw1, vd), vacc1);
            }
            for (; i <= blockEnd - 16; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                vacc0 = Fma.MultiplyAdd(vi, Avx.Multiply(vw, vd), vacc0);
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }
        return (float)sum;
    }

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


    // VecDotQ4_0 — 4-bit block (QK=32)

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
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

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
                var w0 = Avx.Multiply(Vector256.Create(
                    (float)(((qs[i / 2] >> (4 * (i % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 4) / 2] >> (4 * ((i + 4) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 5) / 2] >> (4 * ((i + 5) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 6) / 2] >> (4 * ((i + 6) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 7) / 2] >> (4 * ((i + 7) % 2))) & 0x0F) - 8)
                ), vd);
                var w1 = Avx.Multiply(Vector256.Create(
                    (float)(((qs[(i + 8) / 2] >> (4 * ((i + 8) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 9) / 2] >> (4 * ((i + 9) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 10) / 2] >> (4 * ((i + 10) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 11) / 2] >> (4 * ((i + 11) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 12) / 2] >> (4 * ((i + 12) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 13) / 2] >> (4 * ((i + 13) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 14) / 2] >> (4 * ((i + 14) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 15) / 2] >> (4 * ((i + 15) % 2))) & 0x0F) - 8)
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w0));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i + 8]), w1));
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var w = Avx.Multiply(Vector256.Create(
                    (float)(((qs[i / 2] >> (4 * (i % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 4) / 2] >> (4 * ((i + 4) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 5) / 2] >> (4 * ((i + 5) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 6) / 2] >> (4 * ((i + 6) % 2))) & 0x0F) - 8),
                    (float)(((qs[(i + 7) / 2] >> (4 * ((i + 7) % 2))) & 0x0F) - 8)
                ), vd);
                vacc0 = Avx.Add(vacc0, Avx.Multiply(Vector256.LoadUnsafe(ref pIn[i]), w));
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

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
                float v0 = ((qs[i / 2] >> (4 * (i % 2))) & 0x0F) - 8;
                float v1 = ((qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F) - 8;
                float v2 = ((qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F) - 8;
                float v3 = ((qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F) - 8;
                var vv = Vector128.Create(v0, v1, v2, v3) * Vector128.Create(d);
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum128_Sse(Sse.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }


    // VecDotQ4_1 — 4-bit block with min (QK=32)


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
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

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
            for (; i <= blockEnd - 8; i += 8)
            {
                float v0 = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                float v1 = (qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F;
                float v2 = (qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F;
                float v3 = (qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F;
                float v4 = (qs[(i + 4) / 2] >> (4 * ((i + 4) % 2))) & 0x0F;
                float v5 = (qs[(i + 5) / 2] >> (4 * ((i + 5) % 2))) & 0x0F;
                float v6 = (qs[(i + 6) / 2] >> (4 * ((i + 6) % 2))) & 0x0F;
                float v7 = (qs[(i + 7) / 2] >> (4 * ((i + 7) % 2))) & 0x0F;
                var vv = Avx.Multiply(Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7), vd);
                vv = Avx.Add(vv, vm);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum256_Avx(Avx.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * (q * d + m);
            }
        }
        return (float)sum;
    }

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
                float v0 = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                float v1 = (qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F;
                float v2 = (qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F;
                float v3 = (qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F;
                var vv = Sse.Add(Sse.Multiply(Vector128.Create(v0, v1, v2, v3), vd), vm);
                var vi = Vector128.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum128_Sse(Sse.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * (q * d + m);
            }
        }
        return (float)sum;
    }


    // QuantizedMatMulQ5_0 — fused matmul for q5_0 weights


    public static unsafe void QuantizedMatMulQ5_0_Scalar(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            System.Threading.Tasks.Parallel.For(0, N, col =>
            {
                output[col] = VecDotQ5_0_Scalar(input, rawWeights, col, K);
            });
        }
        else
        {
            System.Threading.Tasks.Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ5_0_Scalar(pInRow, rawWeights, col, K);
            });
        }
    }


    // VecDotQ5_0 — 5-bit block (QK=32)


    public static unsafe float VecDotQ5_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 22;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            uint qh = *(uint*)(block + 2);
            byte* qs = block + 6;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int h4 = ((int)(qh >> i) & 1) << 4;
                int nib = ((i & 1) == 0) ? (qs[i / 2] & 0x0F) : (qs[i / 2] >> 4);
                int q = nib | h4;
                sum += input[b * QK + i] * ((q - 16) * d);
            }
        }
        return (float)sum;
    }


    // VecDotQ5_1 — 5-bit block with min (QK=32)


    public static unsafe float VecDotQ5_1_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 24;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            float m = HalfToFloat_Scalar(*(ushort*)(block + 2));
            uint qh = *(uint*)(block + 4);
            byte* qs = block + 8;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int q = ((qs[i / 2] >> (4 * (i % 2))) & 0x0F) | xh;
                sum += input[b * QK + i] * ((q - 16) * d + m);
            }
        }
        return (float)sum;
    }


    // VecDotQ8_1 — 8-bit block with sum (QK=32)


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
            for (; i <= blockEnd - 24; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i + 8));
                vacc0 = Avx.Add(vacc0, Avx.Multiply(vi0, Avx.Multiply(vw0, vd)));
                vacc1 = Avx.Add(vacc1, Avx.Multiply(vi1, Avx.Multiply(vw1, vd)));
            }
            for (; i <= blockEnd - 16; i += 8)
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

    public static unsafe float VecDotQ8_1_FMA(float* input, byte* rawWeights, int col, int inFeatures)
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
            for (; i <= blockEnd - 24; i += 16)
            {
                var vi0 = Vector256.LoadUnsafe(ref pIn[i]);
                var vi1 = Vector256.LoadUnsafe(ref pIn[i + 8]);
                var vw0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vw1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i + 8));
                vacc0 = Fma.MultiplyAdd(vi0, Avx.Multiply(vw0, vd), vacc0);
                vacc1 = Fma.MultiplyAdd(vi1, Avx.Multiply(vw1, vd), vacc1);
            }
            for (; i <= blockEnd - 16; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                vacc0 = Fma.MultiplyAdd(vi, Avx.Multiply(vw, vd), vacc0);
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

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


    // VecDotQ2K — 2-bit K-quant


    public static unsafe float VecDotQ2K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 16;
            float dSuper = HalfToFloat_Scalar(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 82));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int baseByte = basePos / 4;
                    int isc = (n16 / 128) * 8 + j * 2;
                    float s0 = scales[isc] & 0x0F;
                    float m0 = scales[isc] >> 4;
                    for (int l = 0; l < 16 && basePos + l < blockEnd; l++)
                    {
                        int v = (qs[baseByte + l / 4] >> ((l % 4) * 2)) & 3;
                        sum += input[b * QK_K + basePos + l - colBlockStart] * (s0 * v * dSuper - m0 * minSuper);
                    }

                    float s1 = scales[isc + 1] & 0x0F;
                    float m1 = scales[isc + 1] >> 4;
                    for (int l = 0; l < 16 && basePos + 16 + l < blockEnd; l++)
                    {
                        int v = (qs[baseByte + 4 + l / 4] >> ((l % 4) * 2)) & 3;
                        sum += input[b * QK_K + basePos + 16 + l - colBlockStart] * (s1 * v * dSuper - m1 * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    public static unsafe float VecDotQ2K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 16;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 82));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int baseByte = basePos / 4;
                    int isc = (n16 / 128) * 8 + j * 2;
                    float s0 = scales[isc] & 0x0F;
                    float m0 = scales[isc] >> 4;
                    var vs0 = Vector256.Create(s0 * dSuper);
                    var vm0 = Vector256.Create(m0 * minSuper);

                    int subRem = Math.Min(16, blockEnd - basePos);
                    if (subRem >= 8)
                    {
                        var vacc = Vector256<float>.Zero;
                        int l = 0;
                        for (; l <= subRem - 8; l += 8)
                        {
                            byte b0 = qs[baseByte + l / 4];
                            byte b1 = qs[baseByte + l / 4 + 1];
                            float v0 = b0 & 3;
                            float v1 = (b0 >> 2) & 3;
                            float v2 = (b0 >> 4) & 3;
                            float v3 = (b0 >> 6) & 3;
                            float v4 = b1 & 3;
                            float v5 = (b1 >> 2) & 3;
                            float v6 = (b1 >> 4) & 3;
                            float v7 = (b1 >> 6) & 3;
                            var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                            var vi = Vector256.LoadUnsafe(ref pIn[basePos + l]);
                            var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0), vm0));
                            vacc = Avx.Add(vacc, res);
                        }
                        sum += MathHelpers.HSum256_Avx(vacc);
                        for (; l < subRem; l++)
                        {
                            int pos = b * QK_K + basePos + l - colBlockStart;
                            int v = (qs[baseByte + l / 4] >> ((l % 4) * 2)) & 3;
                            sum += input[pos] * (s0 * v * dSuper - m0 * minSuper);
                        }
                    }
                    else
                    {
                        for (int l = 0; l < subRem; l++)
                        {
                            int pos = b * QK_K + basePos + l - colBlockStart;
                            int v = (qs[baseByte + l / 4] >> ((l % 4) * 2)) & 3;
                            sum += input[pos] * (s0 * v * dSuper - m0 * minSuper);
                        }
                    }

                    float s1 = scales[isc + 1] & 0x0F;
                    float m1 = scales[isc + 1] >> 4;
                    var vs1 = Vector256.Create(s1 * dSuper);
                    var vm1 = Vector256.Create(m1 * minSuper);

                    int bPos1 = basePos + 16;
                    int baseByte1 = bPos1 / 4;
                    int subRem2 = Math.Min(16, blockEnd - bPos1);
                    if (subRem2 >= 8)
                    {
                        var vacc = Vector256<float>.Zero;
                        int l = 0;
                        for (; l <= subRem2 - 8; l += 8)
                        {
                            byte b0 = qs[baseByte1 + l / 4];
                            byte b1 = qs[baseByte1 + l / 4 + 1];
                            float v0 = b0 & 3;
                            float v1 = (b0 >> 2) & 3;
                            float v2 = (b0 >> 4) & 3;
                            float v3 = (b0 >> 6) & 3;
                            float v4 = b1 & 3;
                            float v5 = (b1 >> 2) & 3;
                            float v6 = (b1 >> 4) & 3;
                            float v7 = (b1 >> 6) & 3;
                            var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                            var vi = Vector256.LoadUnsafe(ref pIn[bPos1 + l]);
                            var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1), vm1));
                            vacc = Avx.Add(vacc, res);
                        }
                        sum += MathHelpers.HSum256_Avx(vacc);
                        for (; l < subRem2; l++)
                        {
                            int pos = b * QK_K + bPos1 + l - colBlockStart;
                            int v = (qs[baseByte1 + l / 4] >> ((l % 4) * 2)) & 3;
                            sum += input[pos] * (s1 * v * dSuper - m1 * minSuper);
                        }
                    }
                    else if (subRem2 > 0)
                    {
                        for (int l = 0; l < subRem2; l++)
                        {
                            int pos = b * QK_K + bPos1 + l - colBlockStart;
                            int v = (qs[baseByte1 + l / 4] >> ((l % 4) * 2)) & 3;
                            sum += input[pos] * (s1 * v * dSuper - m1 * minSuper);
                        }
                    }
                }
            }
        }
        return (float)sum;
    }

    public static unsafe float VecDotQ2K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)(startBlock + b) * BLOCK_BYTES;
            byte* scales = block;
            byte* qs = block + 16;
            float dSuper = HalfToFloat_F16C(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_F16C(*(ushort*)(block + 82));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            float* pIn = input + b * QK_K - colBlockStart;
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int baseByte = basePos / 4;
                    int isc = (n16 / 128) * 8 + j * 2;
                    float s0 = scales[isc] & 0x0F;
                    float m0 = scales[isc] >> 4;
                    var vs0 = Vector256.Create(s0 * dSuper);
                    var vm0 = Vector256.Create(m0 * minSuper);

                    int subRem = Math.Min(16, blockEnd - basePos);
                    if (subRem >= 8)
                    {
                        var vacc = Vector256<float>.Zero;
                        int l = 0;
                        for (; l <= subRem - 8; l += 8)
                        {
                            byte b0 = qs[baseByte + l / 4];
                            byte b1 = qs[baseByte + l / 4 + 1];
                            float v0 = b0 & 3;
                            float v1 = (b0 >> 2) & 3;
                            float v2 = (b0 >> 4) & 3;
                            float v3 = (b0 >> 6) & 3;
                            float v4 = b1 & 3;
                            float v5 = (b1 >> 2) & 3;
                            float v6 = (b1 >> 4) & 3;
                            float v7 = (b1 >> 6) & 3;
                            var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                            var vi = Vector256.LoadUnsafe(ref pIn[basePos + l]);
                            var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs0), vm0));
                            vacc = Avx.Add(vacc, res);
                        }
                        sum += MathHelpers.HSum256_Avx(vacc);
                        for (; l < subRem; l++)
                        {
                            int pos = b * QK_K + basePos + l - colBlockStart;
                            int v = (qs[baseByte + l / 4] >> ((l % 4) * 2)) & 3;
                            sum += input[pos] * (s0 * v * dSuper - m0 * minSuper);
                        }
                    }
                    else
                    {
                        for (int l = 0; l < subRem; l++)
                        {
                            int pos = b * QK_K + basePos + l - colBlockStart;
                            int v = (qs[baseByte + l / 4] >> ((l % 4) * 2)) & 3;
                            sum += input[pos] * (s0 * v * dSuper - m0 * minSuper);
                        }
                    }

                    float s1 = scales[isc + 1] & 0x0F;
                    float m1 = scales[isc + 1] >> 4;
                    var vs1 = Vector256.Create(s1 * dSuper);
                    var vm1 = Vector256.Create(m1 * minSuper);

                    int bPos1 = basePos + 16;
                    int baseByte1 = bPos1 / 4;
                    int subRem2 = Math.Min(16, blockEnd - bPos1);
                    if (subRem2 >= 8)
                    {
                        var vacc = Vector256<float>.Zero;
                        int l = 0;
                        for (; l <= subRem2 - 8; l += 8)
                        {
                            byte b0 = qs[baseByte1 + l / 4];
                            byte b1 = qs[baseByte1 + l / 4 + 1];
                            float v0 = b0 & 3;
                            float v1 = (b0 >> 2) & 3;
                            float v2 = (b0 >> 4) & 3;
                            float v3 = (b0 >> 6) & 3;
                            float v4 = b1 & 3;
                            float v5 = (b1 >> 2) & 3;
                            float v6 = (b1 >> 4) & 3;
                            float v7 = (b1 >> 6) & 3;
                            var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                            var vi = Vector256.LoadUnsafe(ref pIn[bPos1 + l]);
                            var res = Avx.Multiply(vi, Avx.Subtract(Avx.Multiply(vv, vs1), vm1));
                            vacc = Avx.Add(vacc, res);
                        }
                        sum += MathHelpers.HSum256_Avx(vacc);
                        for (; l < subRem2; l++)
                        {
                            int pos = b * QK_K + bPos1 + l - colBlockStart;
                            int v = (qs[baseByte1 + l / 4] >> ((l % 4) * 2)) & 3;
                            sum += input[pos] * (s1 * v * dSuper - m1 * minSuper);
                        }
                    }
                    else if (subRem2 > 0)
                    {
                        for (int l = 0; l < subRem2; l++)
                        {
                            int pos = b * QK_K + bPos1 + l - colBlockStart;
                            int v = (qs[baseByte1 + l / 4] >> ((l % 4) * 2)) & 3;
                            sum += input[pos] * (s1 * v * dSuper - m1 * minSuper);
                        }
                    }
                }
            }
        }
        return (float)sum;
    }

    // VecDotQ8K — 8-bit K-quant (QK_K=256)


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

    public static unsafe float VecDotQ8K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
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
                vacc0 = Fma.MultiplyAdd(vi0, Avx.Multiply(vw0, vd), vacc0);
                vacc1 = Fma.MultiplyAdd(vi1, Avx.Multiply(vw1, vd), vacc1);
            }
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                vacc0 = Fma.MultiplyAdd(vi, Avx.Multiply(vw, vd), vacc0);
            }
            sum += MathHelpers.HSum256_Avx(Avx.Add(vacc0, vacc1));
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

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
}
