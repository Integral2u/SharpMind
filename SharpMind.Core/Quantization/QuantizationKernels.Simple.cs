using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Core.Quantization;

internal static partial class QuantizationKernels
{
    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ8_0 — 8-bit block (QK=32)
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ8_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
                sum += input[b * QK + i] * (values[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8_0_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;

            var vacc = Vector256<float>.Zero;
            var vd   = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                var vs = Avx.Multiply(vw, vd);
                vacc = Avx.Add(vacc, Avx.Multiply(vi, vs));
            }
            sum += MathHelpers.HSum256_Avx(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8_0_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;

            var vacc = Vector256<float>.Zero;
            var vd   = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(values + i));
                var vs = Avx.Multiply(vw, vd);
                vacc = Fma.MultiplyAdd(vi, vs, vacc);
            }
            sum += MathHelpers.HSum256_Avx(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8_0_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;

            var vacc = Vector128<float>.Zero;
            var vd   = Vector128.Create(d);
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

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ4_0 — 4-bit block (QK=32)
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ4_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d     = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs    = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ4_0_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs     = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;

            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                float v0 = ((qs[i / 2] >> (4 * (i % 2))) & 0x0F) - 8;
                float v1 = ((qs[(i + 1) / 2] >> (4 * ((i + 1) % 2))) & 0x0F) - 8;
                float v2 = ((qs[(i + 2) / 2] >> (4 * ((i + 2) % 2))) & 0x0F) - 8;
                float v3 = ((qs[(i + 3) / 2] >> (4 * ((i + 3) % 2))) & 0x0F) - 8;
                float v4 = ((qs[(i + 4) / 2] >> (4 * ((i + 4) % 2))) & 0x0F) - 8;
                float v5 = ((qs[(i + 5) / 2] >> (4 * ((i + 5) % 2))) & 0x0F) - 8;
                float v6 = ((qs[(i + 6) / 2] >> (4 * ((i + 6) % 2))) & 0x0F) - 8;
                float v7 = ((qs[(i + 7) / 2] >> (4 * ((i + 7) % 2))) & 0x0F) - 8;
                var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7) * Vector256.Create(d);
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                sum += MathHelpers.HSum256_Avx(Avx.Multiply(vi, vv));
            }
            for (; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += pIn[i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ4_0_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            byte* qs     = block + 2;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;

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

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ4_1 — 4-bit block with min (QK=32)
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ4_1_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d     = HalfToFloat_Scalar(*(ushort*)block);
            float m     = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* qs    = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (qs[i / 2] >> (4 * (i % 2))) & 0x0F;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ4_1_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            float m      = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* qs     = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;
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

    internal static unsafe float VecDotQ4_1_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            float m      = HalfToFloat_Scalar(*(ushort*)(block + 2));
            byte* qs     = block + 4;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;
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

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ5_0 — 5-bit block (QK=32)
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ5_0_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 22;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            uint qh      = *(uint*)(block + 2);
            byte* qs     = block + 6;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int j    = i / 2;
                int h4   = ((int)(qh >> i) & 1) << 4;
                int nib  = (j % 2 == 0) ? (qs[j] & 0x0F) : (qs[j] >> 4);
                int q    = nib | h4;
                sum += input[b * QK + i] * ((q - 16) * d);
            }
        }
        return (float)sum;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ5_1 — 5-bit block with min (QK=32)
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ5_1_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 24;
        const int QK = 32;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            float m      = HalfToFloat_Scalar(*(ushort*)(block + 2));
            uint qh      = *(uint*)(block + 4);
            byte* qs     = block + 8;
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int q  = ((qs[i / 2] >> (4 * (i % 2))) & 0x0F) | xh;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ8_1 — 8-bit block with sum (QK=32)
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ8_1_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* qs    = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
                sum += input[b * QK + i] * (qs[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8_1_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* qs    = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;

            var vacc = Vector256<float>.Zero;
            var vd   = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vs = Avx.Multiply(vw, vd);
                vacc = Avx.Add(vacc, Avx.Multiply(vi, vs));
            }
            sum += MathHelpers.HSum256_Avx(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8_1_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* qs    = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;

            var vacc = Vector256<float>.Zero;
            var vd   = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vs = Avx.Multiply(vw, vd);
                vacc = Fma.MultiplyAdd(vi, vs, vacc);
            }
            sum += MathHelpers.HSum256_Avx(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8_1_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* qs    = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn  = input + b * QK;

            var vacc = Vector128<float>.Zero;
            var vd   = Vector128.Create(d);
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

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ2K — 2-bit K-quant
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ2K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block    = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* scales   = block;
            byte* qs       = block + 16;
            float dSuper   = HalfToFloat_Scalar(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 82));

            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            int qOff = 0;
            for (int n16 = 0; n16 < QK_K; n16 += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int isc = (n16 / 128) * 8 + j * 2;
                    float dl = dSuper * (scales[isc] & 0x0F);
                    float ml = minSuper * (scales[isc] >> 4);
                    for (int l = 0; l < 16 && b * QK_K + n16 + j * 32 + l < b * QK_K + blockEnd; l++)
                    {
                        int v = (qs[qOff + l] >> shift) & 3;
                        sum += input[b * QK_K + n16 + j * 32 + l] * (dl * v - ml);
                    }

                    dl = dSuper * (scales[isc + 1] & 0x0F);
                    ml = minSuper * (scales[isc + 1] >> 4);
                    for (int l = 0; l < 16 && b * QK_K + n16 + j * 32 + 16 + l < b * QK_K + blockEnd; l++)
                    {
                        int v = (qs[qOff + l + 16] >> shift) & 3;
                        sum += input[b * QK_K + n16 + j * 32 + 16 + l] * (dl * v - ml);
                    }
                    shift += 2;
                }
                qOff += 32;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ2K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block    = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* scales   = block;
            byte* qs       = block + 16;
            float dSuper   = HalfToFloat_Scalar(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 82));

            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn  = input + b * QK_K;
            int qOff = 0;
            for (int n16 = 0; n16 < QK_K; n16 += 128)
            {
                int shift = 0;
                int remaining128 = Math.Min(128, blockEnd - n16);
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int isc = (n16 / 128) * 8 + j * 2;
                    float dl = dSuper * (scales[isc] & 0x0F);
                    float ml = minSuper * (scales[isc] >> 4);
                    var vdl = Vector256.Create(dl);
                    var vml = Vector256.Create(ml);

                    int subRem = Math.Min(16, remaining128 - j * 32);
                    if (subRem >= 8)
                    {
                        var vacc = Vector256<float>.Zero;
                        int l = 0;
                        for (; l <= subRem - 8; l += 8)
                        {
                            float v0 = (qs[qOff + l] >> shift) & 3;
                            float v1 = (qs[qOff + l + 1] >> shift) & 3;
                            float v2 = (qs[qOff + l + 2] >> shift) & 3;
                            float v3 = (qs[qOff + l + 3] >> shift) & 3;
                            float v4 = (qs[qOff + l + 4] >> shift) & 3;
                            float v5 = (qs[qOff + l + 5] >> shift) & 3;
                            float v6 = (qs[qOff + l + 6] >> shift) & 3;
                            float v7 = (qs[qOff + l + 7] >> shift) & 3;
                            var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                            var vi = Vector256.LoadUnsafe(ref pIn[n16 + j * 32 + l]);
                            var scaled = Avx.Multiply(vv, vdl);
                            var biased = Avx.Subtract(scaled, vml);
                            vacc = Avx.Add(vacc, Avx.Multiply(vi, biased));
                        }
                        sum += MathHelpers.HSum256_Avx(vacc);
                        for (; l < subRem; l++)
                        {
                            int pos = b * QK_K + n16 + j * 32 + l;
                            int v = (qs[qOff + l] >> shift) & 3;
                            sum += input[pos] * (dl * v - ml);
                        }
                    }
                    else
                    {
                        for (int l = 0; l < subRem; l++)
                        {
                            int pos = b * QK_K + n16 + j * 32 + l;
                            int v = (qs[qOff + l] >> shift) & 3;
                            sum += input[pos] * (dl * v - ml);
                        }
                    }

                    dl = dSuper * (scales[isc + 1] & 0x0F);
                    ml = minSuper * (scales[isc + 1] >> 4);
                    vdl = Vector256.Create(dl);
                    vml = Vector256.Create(ml);

                    int subRem2 = Math.Min(16, remaining128 - j * 32 - 16);
                    if (subRem2 > 0 && subRem2 >= 8)
                    {
                        var vacc = Vector256<float>.Zero;
                        int l = 0;
                        for (; l <= subRem2 - 8; l += 8)
                        {
                            float v0 = (qs[qOff + l + 16] >> shift) & 3;
                            float v1 = (qs[qOff + l + 17] >> shift) & 3;
                            float v2 = (qs[qOff + l + 18] >> shift) & 3;
                            float v3 = (qs[qOff + l + 19] >> shift) & 3;
                            float v4 = (qs[qOff + l + 20] >> shift) & 3;
                            float v5 = (qs[qOff + l + 21] >> shift) & 3;
                            float v6 = (qs[qOff + l + 22] >> shift) & 3;
                            float v7 = (qs[qOff + l + 23] >> shift) & 3;
                            var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                            var vi = Vector256.LoadUnsafe(ref pIn[n16 + j * 32 + 16 + l]);
                            var scaled = Avx.Multiply(vv, vdl);
                            var biased = Avx.Subtract(scaled, vml);
                            vacc = Avx.Add(vacc, Avx.Multiply(vi, biased));
                        }
                        sum += MathHelpers.HSum256_Avx(vacc);
                        for (; l < subRem2; l++)
                        {
                            int pos = b * QK_K + n16 + j * 32 + 16 + l;
                            int v = (qs[qOff + l + 16] >> shift) & 3;
                            sum += input[pos] * (dl * v - ml);
                        }
                    }
                    else
                    {
                        for (int l = 0; l < 16 && n16 + j * 32 + 16 + l < blockEnd; l++)
                        {
                            int pos = b * QK_K + n16 + j * 32 + 16 + l;
                            int v = (qs[qOff + l + 16] >> shift) & 3;
                            sum += input[pos] * (dl * v - ml);
                        }
                    }
                    shift += 2;
                }
                qOff += 32;
            }
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ2K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 84;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block    = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            byte* scales   = block;
            byte* qs       = block + 16;
            float dSuper   = HalfToFloat_Scalar(*(ushort*)(block + 80));
            float minSuper = HalfToFloat_Scalar(*(ushort*)(block + 82));

            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn  = input + b * QK_K;
            int qOff = 0;
            for (int n16 = 0; n16 < QK_K; n16 += 128)
            {
                int shift = 0;
                int remaining128 = Math.Min(128, blockEnd - n16);
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int isc = (n16 / 128) * 8 + j * 2;
                    float dl = dSuper * (scales[isc] & 0x0F);
                    float ml = minSuper * (scales[isc] >> 4);
                    var vdl = Vector256.Create(dl);
                    var vml = Vector256.Create(ml);

                    int subRem = Math.Min(16, remaining128 - j * 32);
                    if (subRem >= 8)
                    {
                        var vacc = Vector256<float>.Zero;
                        int l = 0;
                        for (; l <= subRem - 8; l += 8)
                        {
                            float v0 = (qs[qOff + l] >> shift) & 3;
                            float v1 = (qs[qOff + l + 1] >> shift) & 3;
                            float v2 = (qs[qOff + l + 2] >> shift) & 3;
                            float v3 = (qs[qOff + l + 3] >> shift) & 3;
                            float v4 = (qs[qOff + l + 4] >> shift) & 3;
                            float v5 = (qs[qOff + l + 5] >> shift) & 3;
                            float v6 = (qs[qOff + l + 6] >> shift) & 3;
                            float v7 = (qs[qOff + l + 7] >> shift) & 3;
                            var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                            var vi = Vector256.LoadUnsafe(ref pIn[n16 + j * 32 + l]);
                            var biased = Fma.MultiplyAdd(vi, Avx.Multiply(vv, vdl),
                                Avx.Multiply(vi, Avx.Multiply(vv, vdl)));
                            vacc = Avx.Add(vacc, Avx.Subtract(Avx.Multiply(vi, Avx.Multiply(vv, vdl)),
                                Avx.Multiply(vi, vml)));
                        }
                        sum += MathHelpers.HSum256_Avx(vacc);
                        for (; l < subRem; l++)
                        {
                            int pos = b * QK_K + n16 + j * 32 + l;
                            int v = (qs[qOff + l] >> shift) & 3;
                            sum += input[pos] * (dl * v - ml);
                        }
                    }
                    else
                    {
                        for (int l = 0; l < subRem; l++)
                        {
                            int pos = b * QK_K + n16 + j * 32 + l;
                            int v = (qs[qOff + l] >> shift) & 3;
                            sum += input[pos] * (dl * v - ml);
                        }
                    }

                    dl = dSuper * (scales[isc + 1] & 0x0F);
                    ml = minSuper * (scales[isc + 1] >> 4);
                    vdl = Vector256.Create(dl);
                    vml = Vector256.Create(ml);

                    int subRem2 = Math.Min(16, remaining128 - j * 32 - 16);
                    if (subRem2 > 0 && subRem2 >= 8)
                    {
                        var vacc = Vector256<float>.Zero;
                        int l = 0;
                        for (; l <= subRem2 - 8; l += 8)
                        {
                            float v0 = (qs[qOff + l + 16] >> shift) & 3;
                            float v1 = (qs[qOff + l + 17] >> shift) & 3;
                            float v2 = (qs[qOff + l + 18] >> shift) & 3;
                            float v3 = (qs[qOff + l + 19] >> shift) & 3;
                            float v4 = (qs[qOff + l + 20] >> shift) & 3;
                            float v5 = (qs[qOff + l + 21] >> shift) & 3;
                            float v6 = (qs[qOff + l + 22] >> shift) & 3;
                            float v7 = (qs[qOff + l + 23] >> shift) & 3;
                            var vv = Vector256.Create(v0, v1, v2, v3, v4, v5, v6, v7);
                            var vi = Vector256.LoadUnsafe(ref pIn[n16 + j * 32 + 16 + l]);
                            var biased = Avx.Subtract(Avx.Multiply(vi, Avx.Multiply(vv, vdl)),
                                Avx.Multiply(vi, vml));
                            vacc = Avx.Add(vacc, biased);
                        }
                        sum += MathHelpers.HSum256_Avx(vacc);
                        for (; l < subRem2; l++)
                        {
                            int pos = b * QK_K + n16 + j * 32 + 16 + l;
                            int v = (qs[qOff + l + 16] >> shift) & 3;
                            sum += input[pos] * (dl * v - ml);
                        }
                    }
                    else
                    {
                        for (int l = 0; l < 16 && n16 + j * 32 + 16 + l < blockEnd; l++)
                        {
                            int pos = b * QK_K + n16 + j * 32 + 16 + l;
                            int v = (qs[qOff + l + 16] >> shift) & 3;
                            sum += input[pos] * (dl * v - ml);
                        }
                    }
                    shift += 2;
                }
                qOff += 32;
            }
        }
        return (float)sum;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VecDotQ8K — 8-bit K-quant (QK_K=256)
    // ═══════════════════════════════════════════════════════════════════════

    internal static unsafe float VecDotQ8K_Scalar(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = *(float*)block;
            sbyte* qs    = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            for (int i = 0; i < blockEnd; i++)
                sum += input[b * QK_K + i] * (qs[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8K_AVX2(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = *(float*)block;
            sbyte* qs    = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn  = input + b * QK_K;

            var vacc = Vector256<float>.Zero;
            var vd   = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vs = Avx.Multiply(vw, vd);
                vacc = Avx.Add(vacc, Avx.Multiply(vi, vs));
            }
            sum += MathHelpers.HSum256_Avx(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8K_FMA(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = *(float*)block;
            sbyte* qs    = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn  = input + b * QK_K;

            var vacc = Vector256<float>.Zero;
            var vd   = Vector256.Create(d);
            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                var vi = Vector256.LoadUnsafe(ref pIn[i]);
                var vw = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(qs + i));
                var vs = Avx.Multiply(vw, vd);
                vacc = Fma.MultiplyAdd(vi, vs, vacc);
            }
            sum += MathHelpers.HSum256_Avx(vacc);
            for (; i < blockEnd; i++)
                sum += pIn[i] * (qs[i] * d);
        }
        return (float)sum;
    }

    internal static unsafe float VecDotQ8K_SSE(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 292;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            byte* block  = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d      = *(float*)block;
            sbyte* qs    = (sbyte*)(block + 4);
            int blockEnd = Math.Min(QK_K, inFeatures - b * QK_K);
            float* pIn  = input + b * QK_K;

            var vacc = Vector128<float>.Zero;
            var vd   = Vector128.Create(d);
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
}
