using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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
        // Allocate scores [seqLen, kvLen] on stack for small sequences, heap for large
        var scores = new float[seqLen * kvLen];
        fixed (float* pS = scores)
        {
            // scores = Q @ K^T * scale
            for (int i = 0; i < seqLen; i++)
            {
                float* qi = q + (long)i * headDim;
                for (int j = 0; j < kvLen; j++)
                {
                    if (causal && j > i) { pS[i * kvLen + j] = float.NegativeInfinity; continue; }
                    float* kj = k + (long)j * headDim;
                    var acc = Vector256<float>.Zero;
                    int d = 0;
                    for (; d <= headDim - 8; d += 8)
                        acc = Fma.IsSupported
                            ? Fma.MultiplyAdd(Vector256.LoadUnsafe(ref qi[d]), Vector256.LoadUnsafe(ref kj[d]), acc)
                            : Avx.Add(acc, Avx.Multiply(Vector256.LoadUnsafe(ref qi[d]), Vector256.LoadUnsafe(ref kj[d])));
                    float dot = HSum256(acc);
                    for (; d < headDim; d++) dot += qi[d] * kj[d];
                    pS[i * kvLen + j] = dot * scale;
                }
            }
            // Softmax rows then weight V
            for (int i = 0; i < seqLen; i++)
            {
                SoftmaxInPlace(new Span<float>(pS + i * kvLen, kvLen));
                float* outI = output + (long)i * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kvLen; j++) sum += pS[i * kvLen + j] * v[(long)j * headDim + d];
                    outI[d] = sum;
                }
            }
        }
    }

    internal static unsafe void ScaledDotProductScalar(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal)
    {
        var scores = new float[seqLen * kvLen];
        fixed (float* pS = scores)
        {
            for (int i = 0; i < seqLen; i++)
            {
                float* qi = q + (long)i * headDim;
                for (int j = 0; j < kvLen; j++)
                {
                    if (causal && j > i) { pS[i * kvLen + j] = float.NegativeInfinity; continue; }
                    float* kj = k + (long)j * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qi[d] * kj[d];
                    pS[i * kvLen + j] = dot * scale;
                }
            }
            for (int i = 0; i < seqLen; i++)
            {
                SoftmaxInPlace(new Span<float>(pS + i * kvLen, kvLen));
                float* outI = output + (long)i * headDim;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kvLen; j++) sum += pS[i * kvLen + j] * v[(long)j * headDim + d];
                    outI[d] = sum;
                }
            }
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float HSum256(Vector256<float> v)
    {
        var lo = Avx.ExtractVector128(v, 0);
        var hi = Avx.ExtractVector128(v, 1);
        var s = Sse.Add(lo, hi);
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        return Sse.AddScalar(s, Sse.Shuffle(s, s, 1)).ToScalar();
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
