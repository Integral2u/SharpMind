using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Buffers;
using SharpMind.Core;

namespace SharpMind.Model.Layers.Attention;

// ─────────────────────────────────────────────────────────────────────────────
// Attention kernels — pure static, one unconditional path each
// ─────────────────────────────────────────────────────────────────────────────

internal static class AttentionKernels
{
    /// <summary>
    /// Scaled dot-product attention inner kernel.
    /// Q: [SeqLen, HeadDim]  K: [KvLen, HeadDim]  V: [KvLen, HeadDim]
    /// Out: [SeqLen, HeadDim]
    /// Out: [SeqLen, HeadDim]
    /// Causal mask applied when <paramref name="causal"/> is true.
    /// </summary>
    internal static unsafe void ScaledDotProductAVX2(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal)
    {
        float[] rented = ArrayPool<float>.Shared.Rent(kvLen);
        try
        {
            Span<float> scoreRow = rented.AsSpan(0, kvLen);
            // During KV-cache decode, the query offset in the full KV sequence
            // is kvLen - seqLen (the number of tokens before the current step).
            int queryBase = causal ? kvLen - seqLen : 0;
            for (int i = 0; i < seqLen; i++)
            {
                float* qi = q + (long)i * headDim;
                int absQPos = queryBase + i;
                for (int j = 0; j < kvLen; j++)
                {
                    if (causal && j > absQPos)
                    {
                        scoreRow[j] = float.NegativeInfinity;
                        continue;
                    }
                    float* kj = k + (long)j * headDim;
                    var acc = Vector256<float>.Zero;
                    int d = 0;
                    for (; d <= headDim - 8; d += 8)
                        acc = Fma.IsSupported
                            ? Fma.MultiplyAdd(Vector256.LoadUnsafe(ref qi[d]), Vector256.LoadUnsafe(ref kj[d]), acc)
                            : Avx.Add(acc, Avx.Multiply(Vector256.LoadUnsafe(ref qi[d]), Vector256.LoadUnsafe(ref kj[d])));
                    float dot = MathHelpers.HSum256_Avx(acc);
                    for (; d < headDim; d++) dot += qi[d] * kj[d];
                    scoreRow[j] = dot * scale;
                }

                SoftmaxInPlace(scoreRow);
                float* outI = output + (long)i * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kvLen; j++)
                        sum += scoreRow[j] * v[(long)j * headDim + d];
                    outI[d] = sum;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    internal static unsafe void ScaledDotProductScalar(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal)
    {
        float[] rented = ArrayPool<float>.Shared.Rent(kvLen);
        try
        {
            Span<float> scoreRow = rented.AsSpan(0, kvLen);
            int queryBase = causal ? kvLen - seqLen : 0;
            for (int i = 0; i < seqLen; i++)
            {
                float* qi = q + (long)i * headDim;
                int absQPos = queryBase + i;
                for (int j = 0; j < kvLen; j++)
                {
                    if (causal && j > absQPos)
                    {
                        scoreRow[j] = float.NegativeInfinity;
                        continue;
                    }
                    float* kj = k + (long)j * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qi[d] * kj[d];
                    scoreRow[j] = dot * scale;
                }

                SoftmaxInPlace(scoreRow);
                float* outI = output + (long)i * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kvLen; j++)
                        sum += scoreRow[j] * v[(long)j * headDim + d];
                    outI[d] = sum;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    private static unsafe void SoftmaxInPlace(Span<float> row)
    {
        float max = row[0];
        foreach (float v in row) if (v > max) max = v;
        float sum = 0f;
        for (int i = 0; i < row.Length; i++) { row[i] = MathF.Exp(row[i] - max); sum += row[i]; }
        float inv = 1f / sum;
        for (int i = 0; i < row.Length; i++) row[i] *= inv;
    }
}
