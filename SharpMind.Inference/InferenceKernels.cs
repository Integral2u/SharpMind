using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;

namespace SharpMind.Inference;

// ─────────────────────────────────────────────────────────────────────────────
// Attention kernels — pure static paths, no runtime capability checks
// ─────────────────────────────────────────────────────────────────────────────

internal static class InferenceKernels
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Inference)}.{nameof(InferenceKernels)}";

    // ── Standard decode attention ─────────────────────────────────────────
    // Q: [1, HeadDim]   K: [CacheLen, HeadDim]   V: [CacheLen, HeadDim]
    // Out: [HeadDim]
    // Single-vector query attending to full KV cache — no causal mask needed
    // (all cached tokens are already in the past)

    internal static unsafe void DecodeAttention_Standard_AVX2(
        float* q, float* k, float* v, float* output,
        int cacheLen, int headDim, float scale)
    {
        float[]? rentedScores = cacheLen > 256 ? ArrayPool<float>.Shared.Rent(cacheLen) : null;
        Span<float> scores = rentedScores is not null
            ? rentedScores.AsSpan(0, cacheLen)
            : stackalloc float[cacheLen];
        fixed (float* pS = scores)
        {
            for (int j = 0; j < cacheLen; j++)
            {
                float* kj  = k + (long)j * headDim;
                var    acc = Vector256<float>.Zero;
                int    d   = 0;
                for (; d <= headDim - 8; d += 8)
                    acc = Avx.Add(acc, Avx.Multiply(Vector256.LoadUnsafe(ref q[d]),
                                                     Vector256.LoadUnsafe(ref kj[d])));
                float dot = MathHelpers.HSum256_Avx(acc);
                for (; d < headDim; d++) dot += q[d] * kj[d];
                pS[j] = dot * scale;
            }
            SoftmaxInPlace(scores);
            for (int d = 0; d < headDim; d++)
            {
                float sum = 0f;
                for (int j = 0; j < cacheLen; j++) sum += pS[j] * v[(long)j * headDim + d];
                output[d] = sum;
            }
        }
        if (rentedScores is not null) ArrayPool<float>.Shared.Return(rentedScores);
    }

    internal static unsafe void DecodeAttention_Standard_FMA(
        float* q, float* k, float* v, float* output,
        int cacheLen, int headDim, float scale)
    {
        float[]? rentedScores = cacheLen > 256 ? ArrayPool<float>.Shared.Rent(cacheLen) : null;
        Span<float> scores = rentedScores is not null
            ? rentedScores.AsSpan(0, cacheLen)
            : stackalloc float[cacheLen];
        fixed (float* pS = scores)
        {
            for (int j = 0; j < cacheLen; j++)
            {
                float* kj  = k + (long)j * headDim;
                var    acc = Vector256<float>.Zero;
                int    d   = 0;
                for (; d <= headDim - 8; d += 8)
                    acc = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref q[d]),
                                          Vector256.LoadUnsafe(ref kj[d]), acc);
                float dot = MathHelpers.HSum256_Avx(acc);
                for (; d < headDim; d++) dot += q[d] * kj[d];
                pS[j] = dot * scale;
            }
            SoftmaxInPlace(scores);
            for (int d = 0; d < headDim; d++)
            {
                float sum = 0f;
                for (int j = 0; j < cacheLen; j++) sum += pS[j] * v[(long)j * headDim + d];
                output[d] = sum;
            }
        }
        if (rentedScores is not null) ArrayPool<float>.Shared.Return(rentedScores);
    }

    internal static unsafe void DecodeAttention_Standard_Scalar(
        float* q, float* k, float* v, float* output,
        int cacheLen, int headDim, float scale)
    {
        float[]? rentedScores = cacheLen > 256 ? ArrayPool<float>.Shared.Rent(cacheLen) : null;
        Span<float> scores = rentedScores is not null
            ? rentedScores.AsSpan(0, cacheLen)
            : stackalloc float[cacheLen];

        for (int j = 0; j < cacheLen; j++)
        {
            float dot = 0f;
            float* kj = k + (long)j * headDim;
            for (int d = 0; d < headDim; d++) dot += q[d] * kj[d];
            scores[j] = dot * scale;
        }
        SoftmaxInPlace(scores);
        for (int d = 0; d < headDim; d++)
        {
            float sum = 0f;
            for (int j = 0; j < cacheLen; j++) sum += scores[j] * v[(long)j * headDim + d];
            output[d] = sum;
        }
        if (rentedScores is not null) ArrayPool<float>.Shared.Return(rentedScores);
    }

    // ── Flash decode attention — tiled to avoid large score buffer ────────
    // Uses online softmax (Milakov & Gimelshein, 2018) so the score buffer
    // is O(tile) rather than O(CacheLen). Critical for long-context inference.

    internal static unsafe void DecodeAttention_Flash_AVX2(
        float* q, float* k, float* v, float* output,
        int cacheLen, int headDim, float scale)
        => DecodeAttention_Flash_Core(q, k, v, output, cacheLen, headDim, scale, avx2: true, fma: false);

    internal static unsafe void DecodeAttention_Flash_FMA(
        float* q, float* k, float* v, float* output,
        int cacheLen, int headDim, float scale)
        => DecodeAttention_Flash_Core(q, k, v, output, cacheLen, headDim, scale, avx2: true, fma: true);

    internal static unsafe void DecodeAttention_Flash_Scalar(
        float* q, float* k, float* v, float* output,
        int cacheLen, int headDim, float scale)
        => DecodeAttention_Flash_Core(q, k, v, output, cacheLen, headDim, scale, avx2: false, fma: false);

    private static unsafe void DecodeAttention_Flash_Core(
    float* q, float* k, float* v, float* output,
    int cacheLen, int headDim, float scale, bool avx2, bool fma)
    {
        const int MaxHeadDim = 256;
        if ((uint)headDim > MaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds MaxHeadDim {MaxHeadDim}.");

        const int TileSize = 64;

        float mMax = float.NegativeInfinity;
        float lSum = 0f;

        float* pO = stackalloc float[MaxHeadDim];
        for (int d = 0; d < headDim; d++) pO[d] = 0f;

        float* tileScores = stackalloc float[TileSize];

        for (int start = 0; start < cacheLen; start += TileSize)
        {
            int end = Math.Min(start + TileSize, cacheLen);
            int tileLen = end - start;

            float tileMax = float.NegativeInfinity;
            for (int j = start; j < end; j++)
            {
                float* kj = k + (long)j * headDim;
                float dot = 0f;
                if (fma)
                {
                    var acc = Vector256<float>.Zero;
                    int d = 0;
                    for (; d <= headDim - 8; d += 8)
                        acc = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref q[d]),
                                              Vector256.LoadUnsafe(ref kj[d]), acc);
                    dot = MathHelpers.HSum256_Avx(acc);
                    for (; d < headDim; d++) dot += q[d] * kj[d];
                }
                else if (avx2)
                {
                    var acc = Vector256<float>.Zero;
                    int d = 0;
                    for (; d <= headDim - 8; d += 8)
                        acc = Avx.Add(acc, Avx.Multiply(
                            Vector256.LoadUnsafe(ref q[d]),
                            Vector256.LoadUnsafe(ref kj[d])));
                    dot = MathHelpers.HSum256_Avx(acc);
                    for (; d < headDim; d++) dot += q[d] * kj[d];
                }
                else
                {
                    for (int d = 0; d < headDim; d++) dot += q[d] * kj[d];
                }
                dot *= scale;
                tileScores[j - start] = dot;
                if (dot > tileMax) tileMax = dot;
            }

            float newMax = MathF.Max(mMax, tileMax);
            float scaleOld = MathF.Exp(mMax - newMax);
            float scaleNew = 0f;
            for (int i = 0; i < tileLen; i++)
            {
                tileScores[i] = MathF.Exp(tileScores[i] - newMax);
                scaleNew += tileScores[i];
            }

            float newL = scaleOld * lSum + scaleNew;
            for (int d = 0; d < headDim; d++)
                pO[d] *= scaleOld;
            for (int i = 0; i < tileLen; i++)
            {
                float* vj = v + (long)(start + i) * headDim;
                for (int d = 0; d < headDim; d++)
                    pO[d] += tileScores[i] * vj[d];
            }
            mMax = newMax;
            lSum = newL;
        }

        for (int d = 0; d < headDim; d++)
            output[d] = lSum > 0f ? pO[d] / lSum : 0f;
    }

    // ── INT8 quantized matmul ─────────────────────────────────────────────
    // Weights stored as INT8; dequantized to float before multiply.
    // Reduces memory bandwidth by ~4× for weight-only quantization.
    // Scale and zero-point per output channel (symmetric or asymmetric).

    internal static unsafe void QuantMatMul_Int8(
        float* input, sbyte* weights, float* output,
        float* scales,    // [OutFeatures] dequantization scale per channel
        int inFeatures, int outFeatures)
    {
        for (int o = 0; o < outFeatures; o++)
        {
            float scale = scales[o];
            sbyte* row  = weights + (long)o * inFeatures;
            float  sum  = 0f;
            for (int i = 0; i < inFeatures; i++)
                sum += input[i] * (row[i] * scale);
            output[o] = sum;
        }
    }

    internal static unsafe void QuantMatMul_FP32(
        float* input, float* weights, float* output,
        float* scales, int inFeatures, int outFeatures)
    {
        // Standard float32 matmul — scales are identity, used via same slot
        for (int o = 0; o < outFeatures; o++)
        {
            float* row = weights + (long)o * inFeatures;
            float  sum = 0f;
            for (int i = 0; i < inFeatures; i++) sum += input[i] * row[i];
            output[o] = sum;
        }
    }

    private static void SoftmaxInPlace(Span<float> x)
    {
        float max = x[0];
        foreach (float v in x) if (v > max) max = v;
        float sum = 0f;
        for (int i = 0; i < x.Length; i++) { x[i] = MathF.Exp(x[i] - max); sum += x[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < x.Length; i++) x[i] *= inv;
    }
}
