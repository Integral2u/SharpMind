using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;
using static SharpMind.Core.Quantization.QuantizationKernels;

namespace SharpMind.Model.Layers.Attention;

// Attention kernels, pure static, one unconditional path each

public static class AttentionKernels
{
    /// <summary>Reusable score buffer per thread, avoids per-call ArrayPool.Rent.</summary>
    [ThreadStatic]
    private static float[]? t_ScoreScratch;

    /// <summary>
    /// Scaled dot-product attention inner kernel.
    /// Q: [SeqLen, HeadDim]  K: [KvLen, HeadDim]  V: [KvLen, HeadDim]
    /// Out: [SeqLen, HeadDim]
    /// Uses online softmax (Milakov & Gimelshein, 2018) integrated with the
    /// score-computation and V-weighted accumulation loops, 2 passes over KV
    /// instead of 3 softmax passes + 1 V-accumulation pass.
    /// </summary>
    public static unsafe void ScaledDotProductAVX2(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        EnsureScoreBuffer(kvLen);
        fixed (float* pRow = &t_ScoreScratch![0])
        {
            float* scoreRow = pRow;
            int queryBase = causal ? kvLen - seqLen : 0;
            for (int i = 0; i < seqLen; i++)
            {
                float* qi = q + (long)i * qStride;
                int absQPos = queryBase + i;
                int kvOffset = 0;
                int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
                if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

                // Pass 1: compute scores + online softmax statistics
                float max = float.NegativeInfinity;
                float lSum = 0f;
                for (int j = 0; j < effKvLen; j++)
                {
                    float* kj = k + (long)(kvOffset + j) * headDim;
                    var acc = Vector256<float>.Zero;
                    int d = 0;
                    for (; d <= headDim - 8; d += 8)
                        acc = Avx.Add(acc, Avx.Multiply(
                            Vector256.LoadUnsafe(ref qi[d]),
                            Vector256.LoadUnsafe(ref kj[d])));
                    float dot = MathHelpers.HSum256_Avx(acc);
                    for (; d < headDim; d++) dot += qi[d] * kj[d];
                    float score = dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    scoreRow[j] = score;
                    float oldMax = max;
                    max = Math.Max(max, score);
                    lSum = lSum * MathF.Exp(oldMax - max) + MathF.Exp(score - max);
                }

                // Pass 2: normalize + accumulate V-weighted output (vectorized SAXPY)
                float* outI = output + (long)i * oStride;
                if (lSum > 0f)
                {
                    float invSum = 1f / lSum;
                    for (int d = 0; d < headDim; d++) outI[d] = 0f;
                    for (int j = 0; j < effKvLen; j++)
                    {
                        float sm = MathF.Exp(scoreRow[j] - max) * invSum;
                        float* vj = v + (long)(kvOffset + j) * headDim;
                        var vSm = Vector256.Create(sm);
                        int d = 0;
                        for (; d <= headDim - 8; d += 8)
                            Vector256.StoreUnsafe(
                                Avx.Add(
                                    Avx.Multiply(Vector256.LoadUnsafe(ref vj[d]), vSm),
                                    Vector256.LoadUnsafe(ref outI[d])),
                                ref outI[d]);
                        for (; d < headDim; d++)
                            outI[d] += sm * vj[d];
                    }
                }
                else
                {
                    for (int d = 0; d < headDim; d++) outI[d] = 0f;
                }
            }
        }
    }

    public static unsafe void ScaledDotProductFMA(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        EnsureScoreBuffer(kvLen);
        fixed (float* pRow = &t_ScoreScratch![0])
        {
            float* scoreRow = pRow;
            int queryBase = causal ? kvLen - seqLen : 0;
            for (int i = 0; i < seqLen; i++)
            {
                float* qi = q + (long)i * qStride;
                int absQPos = queryBase + i;
                int kvOffset = 0;
                int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
                if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

                float max = float.NegativeInfinity;
                float lSum = 0f;
                for (int j = 0; j < effKvLen; j++)
                {
                    float* kj = k + (long)(kvOffset + j) * headDim;
                    var acc = Vector256<float>.Zero;
                    int d = 0;
                    for (; d <= headDim - 8; d += 8)
                        acc = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref qi[d]),
                                              Vector256.LoadUnsafe(ref kj[d]), acc);
                    float dot = MathHelpers.HSum256_Avx(acc);
                    for (; d < headDim; d++) dot += qi[d] * kj[d];
                    float score = dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    scoreRow[j] = score;
                    float oldMax = max;
                    max = Math.Max(max, score);
                    lSum = lSum * MathF.Exp(oldMax - max) + MathF.Exp(score - max);
                }

                float* outI = output + (long)i * oStride;
                if (lSum > 0f)
                {
                    float invSum = 1f / lSum;
                    for (int d = 0; d < headDim; d++) outI[d] = 0f;
                    for (int j = 0; j < effKvLen; j++)
                    {
                        float sm = MathF.Exp(scoreRow[j] - max) * invSum;
                        float* vj = v + (long)(kvOffset + j) * headDim;
                        var vSm = Vector256.Create(sm);
                        int d = 0;
                        for (; d <= headDim - 8; d += 8)
                            Vector256.StoreUnsafe(
                                Fma.MultiplyAdd(
                                    Vector256.LoadUnsafe(ref vj[d]), vSm,
                                    Vector256.LoadUnsafe(ref outI[d])),
                                ref outI[d]);
                        for (; d < headDim; d++)
                            outI[d] += sm * vj[d];
                    }
                }
                else
                {
                    for (int d = 0; d < headDim; d++) outI[d] = 0f;
                }
            }
        }
    }

    public static unsafe void ScaledDotProductScalar(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        EnsureScoreBuffer(kvLen);
        fixed (float* pRow = &t_ScoreScratch![0])
        {
            float* scoreRow = pRow;
            int queryBase = causal ? kvLen - seqLen : 0;
            for (int i = 0; i < seqLen; i++)
            {
                float* qi = q + (long)i * qStride;
                int absQPos = queryBase + i;
                int kvOffset = 0;
                int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
                if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

                float max = float.NegativeInfinity;
                float lSum = 0f;
                for (int j = 0; j < effKvLen; j++)
                {
                    float* kj = k + (long)(kvOffset + j) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qi[d] * kj[d];
                    float score = dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    scoreRow[j] = score;
                    float oldMax = max;
                    max = Math.Max(max, score);
                    lSum = lSum * MathF.Exp(oldMax - max) + MathF.Exp(score - max);
                }

                float* outI = output + (long)i * oStride;
                if (lSum > 0f)
                {
                    float invSum = 1f / lSum;
                    for (int d = 0; d < headDim; d++) outI[d] = 0f;
                    for (int j = 0; j < effKvLen; j++)
                    {
                        float sm = MathF.Exp(scoreRow[j] - max) * invSum;
                        float* vj = v + (long)(kvOffset + j) * headDim;
                        for (int d = 0; d < headDim; d++)
                            outI[d] += sm * vj[d];
                    }
                }
                else
                {
                    for (int d = 0; d < headDim; d++) outI[d] = 0f;
                }
            }
        }
    }

    private static void EnsureScoreBuffer(int kvLen)
    {
        if (t_ScoreScratch is null || t_ScoreScratch.Length < kvLen)
            t_ScoreScratch = GC.AllocateUninitializedArray<float>(
                Math.Max(kvLen, 64));
    }

    // Flash attention
    // Online softmax (Milakov & Gimelshein, 2018) tiles KV positions to
    // avoid O(kvLen) score buffer. Essential for long-context inference.

    private const int FlashTileSize = 64;
    private const int FlashMaxHeadDim = 256;

    public static unsafe void ScaledDotProductFlashAVX2(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        float* tileScores = stackalloc float[FlashTileSize];
        int queryBase = causal ? kvLen - seqLen : 0;

        for (int i = 0; i < seqLen; i++)
        {
            float* qi = q + (long)i * qStride;
            int absQPos = queryBase + i;
            int kvOffset = 0;
            int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
            if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

            float mMax = float.NegativeInfinity;
            float lSum = 0f;
            float* pO = output + (long)i * oStride;
            for (int d = 0; d < headDim; d++) pO[d] = 0f;

            for (int start = 0; start < effKvLen; start += FlashTileSize)
            {
                int end = Math.Min(start + FlashTileSize, effKvLen);
                int tileLen = end - start;

                float tileMax = float.NegativeInfinity;
                for (int j = start; j < end; j++)
                {
                    float* kj = k + (long)(kvOffset + j) * headDim;
                    var acc = Vector256<float>.Zero;
                    int d = 0;
                    for (; d <= headDim - 8; d += 8)
                        acc = Avx.Add(acc, Avx.Multiply(
                            Vector256.LoadUnsafe(ref qi[d]),
                            Vector256.LoadUnsafe(ref kj[d])));
                    float dot = MathHelpers.HSum256_Avx(acc);
                    for (; d < headDim; d++) dot += qi[d] * kj[d];
                    dot = dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    tileScores[j - start] = dot;
                    if (dot > tileMax) tileMax = dot;
                }

                float newMax = MathF.Max(mMax, tileMax);
                float scaleOld = MathF.Exp(mMax - newMax);
                float scaleNew = 0f;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - newMax);
                    scaleNew += tileScores[t];
                }

                float newL = scaleOld * lSum + scaleNew;
                var vScaleOld = Vector256.Create(scaleOld);
                int d0 = 0;
                for (; d0 <= headDim - 8; d0 += 8)
                    Vector256.StoreUnsafe(
                        Avx.Multiply(Vector256.LoadUnsafe(ref pO[d0]), vScaleOld),
                        ref pO[d0]);
                for (; d0 < headDim; d0++)
                    pO[d0] *= scaleOld;
                for (int t = 0; t < tileLen; t++)
                {
                    float* vj = v + (long)(kvOffset + start + t) * headDim;
                    var vSm = Vector256.Create(tileScores[t]);
                    int d = 0;
                    for (; d <= headDim - 8; d += 8)
                        Vector256.StoreUnsafe(
                            Avx.Add(Vector256.LoadUnsafe(ref pO[d]),
                                    Avx.Multiply(Vector256.LoadUnsafe(ref vj[d]), vSm)),
                            ref pO[d]);
                    for (; d < headDim; d++)
                        pO[d] += tileScores[t] * vj[d];
                }
                mMax = newMax;
                lSum = newL;
            }

            if (lSum > 0f)
            {
                var vLSum = Vector256.Create(lSum);
                int d = 0;
                for (; d <= headDim - 8; d += 8)
                    Vector256.StoreUnsafe(
                        Avx.Divide(Vector256.LoadUnsafe(ref pO[d]), vLSum),
                        ref pO[d]);
                for (; d < headDim; d++) pO[d] /= lSum;
            }
            else
                for (int d = 0; d < headDim; d++) pO[d] = 0f;
        }
    }

    /// <summary>
    /// FMA attention for one head. Scores come eight keys per pass: one query load feeds eight
    /// accumulators whose key loads fold into the FMA memory operands, so the key strip streams
    /// eight rows at a time. The softmax runs over the whole row with <see cref="Vector256.Exp"/>
    /// (the online form's per-key MathF.Exp cost ~1.8 ns an element against ~0.4), and values
    /// accumulate four rows per pass. softmax(s)_j = exp(s_j - max s) / sum, as before up to rounding.
    /// Techniques from pulsarforge (MIT, github.com/siris9476/pulsarforge: attn_q8_group,
    /// dotq8i_x8, axpyq8_x4), here in float.
    /// </summary>
    public static unsafe void ScaledDotProductFlashFMA(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        EnsureScoreBuffer(kvLen);
        fixed (float* scores = t_ScoreScratch!)
        {
            int queryBase = causal ? kvLen - seqLen : 0;
            int hd8 = headDim & ~7;
            var vScale = Vector256.Create(scale);
            for (int i = 0; i < seqLen; i++)
            {
                float* qi = q + (long)i * qStride;
                int absQPos = queryBase + i;
                int kvOffset = 0;
                int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
                if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

                float* pO = output + (long)i * oStride;
                new Span<float>(pO, headDim).Clear();
                if (effKvLen <= 0) continue;
                float* kb = k + (long)kvOffset * headDim;
                float* vb = v + (long)kvOffset * headDim;

                int j = 0;
                for (; j + 8 <= effKvLen; j += 8)
                    DotEightKeys(qi, kb + (long)j * headDim, headDim, scores + j);
                for (; j < effKvLen; j++)
                    scores[j] = DotOneKey(qi, kb + (long)j * headDim, headDim);

                float max = float.NegativeInfinity;
                if (alibiSlope == 0f)
                {
                    var vMax = Vector256.Create(float.NegativeInfinity);
                    for (j = 0; j + 8 <= effKvLen; j += 8)
                    {
                        var s = Avx.Multiply(Avx.LoadVector256(scores + j), vScale);
                        Avx.Store(scores + j, s);
                        vMax = Avx.Max(vMax, s);
                    }
                    var m4 = Sse.Max(vMax.GetLower(), vMax.GetUpper());
                    m4 = Sse.Max(m4, Sse.MoveHighToLow(m4, m4));
                    max = MathF.Max(m4.ToScalar(), m4.GetElement(1));
                    for (; j < effKvLen; j++)
                    {
                        scores[j] *= scale;
                        if (scores[j] > max) max = scores[j];
                    }
                }
                else
                {
                    for (j = 0; j < effKvLen; j++)
                    {
                        float s = scores[j] * scale - alibiSlope * (absQPos - kvOffset - j);
                        scores[j] = s;
                        if (s > max) max = s;
                    }
                }

                var vMaxScore = Vector256.Create(max);
                var expSum = Vector256<float>.Zero;
                for (j = 0; j + 8 <= effKvLen; j += 8)
                {
                    var e = Vector256.Exp(Avx.Subtract(Avx.LoadVector256(scores + j), vMaxScore));
                    Avx.Store(scores + j, e);
                    expSum = Avx.Add(expSum, e);
                }
                float sum = Vector256.Sum(expSum);
                for (; j < effKvLen; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - max);
                    sum += scores[j];
                }
                if (!(sum > 0f)) continue;   // NaN scores: leave the row zero rather than spread NaN
                float inv = 1f / sum;

                for (j = 0; j + 4 <= effKvLen; j += 4)
                {
                    float w0 = scores[j] * inv, w1 = scores[j + 1] * inv, w2 = scores[j + 2] * inv, w3 = scores[j + 3] * inv;
                    float* v0 = vb + (long)j * headDim, v1 = v0 + headDim, v2 = v1 + headDim, v3 = v2 + headDim;
                    var c0 = Vector256.Create(w0);
                    var c1 = Vector256.Create(w1);
                    var c2 = Vector256.Create(w2);
                    var c3 = Vector256.Create(w3);
                    int d = 0;
                    for (; d < hd8; d += 8)
                    {
                        var o = Avx.LoadVector256(pO + d);
                        o = Fma.MultiplyAdd(Avx.LoadVector256(v0 + d), c0, o);
                        o = Fma.MultiplyAdd(Avx.LoadVector256(v1 + d), c1, o);
                        o = Fma.MultiplyAdd(Avx.LoadVector256(v2 + d), c2, o);
                        o = Fma.MultiplyAdd(Avx.LoadVector256(v3 + d), c3, o);
                        Avx.Store(pO + d, o);
                    }
                    for (; d < headDim; d++)
                        pO[d] += w0 * v0[d] + w1 * v1[d] + w2 * v2[d] + w3 * v3[d];
                }
                for (; j < effKvLen; j++)
                {
                    float w = scores[j] * inv;
                    float* vj = vb + (long)j * headDim;
                    var c = Vector256.Create(w);
                    int d = 0;
                    for (; d < hd8; d += 8)
                        Avx.Store(pO + d, Fma.MultiplyAdd(Avx.LoadVector256(vj + d), c, Avx.LoadVector256(pO + d)));
                    for (; d < headDim; d++)
                        pO[d] += w * vj[d];
                }
            }
        }
    }

    /// <summary>Eight query·key dot products; each key load folds into its FMA's memory operand.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static unsafe void DotEightKeys(float* q, float* k0, int headDim, float* out8)
    {
        float* k1 = k0 + headDim, k2 = k1 + headDim, k3 = k2 + headDim;
        float* k4 = k3 + headDim, k5 = k4 + headDim, k6 = k5 + headDim, k7 = k6 + headDim;
        var a0 = Vector256<float>.Zero; var a1 = a0; var a2 = a0; var a3 = a0;
        var a4 = a0; var a5 = a0; var a6 = a0; var a7 = a0;
        int d = 0;
        for (; d + 8 <= headDim; d += 8)
        {
            var vq = Avx.LoadVector256(q + d);
            a0 = Fma.MultiplyAdd(vq, Avx.LoadVector256(k0 + d), a0);
            a1 = Fma.MultiplyAdd(vq, Avx.LoadVector256(k1 + d), a1);
            a2 = Fma.MultiplyAdd(vq, Avx.LoadVector256(k2 + d), a2);
            a3 = Fma.MultiplyAdd(vq, Avx.LoadVector256(k3 + d), a3);
            a4 = Fma.MultiplyAdd(vq, Avx.LoadVector256(k4 + d), a4);
            a5 = Fma.MultiplyAdd(vq, Avx.LoadVector256(k5 + d), a5);
            a6 = Fma.MultiplyAdd(vq, Avx.LoadVector256(k6 + d), a6);
            a7 = Fma.MultiplyAdd(vq, Avx.LoadVector256(k7 + d), a7);
        }
        out8[0] = Vector256.Sum(a0); out8[1] = Vector256.Sum(a1); out8[2] = Vector256.Sum(a2); out8[3] = Vector256.Sum(a3);
        out8[4] = Vector256.Sum(a4); out8[5] = Vector256.Sum(a5); out8[6] = Vector256.Sum(a6); out8[7] = Vector256.Sum(a7);
        for (; d < headDim; d++)
        {
            float qd = q[d];
            out8[0] += qd * k0[d]; out8[1] += qd * k1[d]; out8[2] += qd * k2[d]; out8[3] += qd * k3[d];
            out8[4] += qd * k4[d]; out8[5] += qd * k5[d]; out8[6] += qd * k6[d]; out8[7] += qd * k7[d];
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotOneKey(float* q, float* kj, int headDim)
    {
        var acc = Vector256<float>.Zero;
        int d = 0;
        for (; d + 8 <= headDim; d += 8)
            acc = Fma.MultiplyAdd(Avx.LoadVector256(q + d), Avx.LoadVector256(kj + d), acc);
        float dot = Vector256.Sum(acc);
        for (; d < headDim; d++) dot += q[d] * kj[d];
        return dot;
    }

    public static unsafe void ScaledDotProductFlashScalar(
        float* q, float* k, float* v, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        float* tileScores = stackalloc float[FlashTileSize];
        int queryBase = causal ? kvLen - seqLen : 0;

        for (int i = 0; i < seqLen; i++)
        {
            float* qi = q + (long)i * qStride;
            int absQPos = queryBase + i;
            int kvOffset = 0;
            int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
            if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

            float mMax = float.NegativeInfinity;
            float lSum = 0f;
            float* pO = output + (long)i * oStride;
            for (int d = 0; d < headDim; d++) pO[d] = 0f;

            for (int start = 0; start < effKvLen; start += FlashTileSize)
            {
                int end = Math.Min(start + FlashTileSize, effKvLen);
                int tileLen = end - start;

                float tileMax = float.NegativeInfinity;
                for (int j = start; j < end; j++)
                {
                    float* kj = k + (long)(kvOffset + j) * headDim;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += qi[d] * kj[d];
                    dot = dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    tileScores[j - start] = dot;
                    if (dot > tileMax) tileMax = dot;
                }

                float newMax = MathF.Max(mMax, tileMax);
                float scaleOld = MathF.Exp(mMax - newMax);
                float scaleNew = 0f;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - newMax);
                    scaleNew += tileScores[t];
                }

                float newL = scaleOld * lSum + scaleNew;
                for (int d = 0; d < headDim; d++)
                    pO[d] *= scaleOld;
                for (int t = 0; t < tileLen; t++)
                {
                    float* vj = v + (long)(kvOffset + start + t) * headDim;
                    for (int d = 0; d < headDim; d++)
                        pO[d] += tileScores[t] * vj[d];
                }
                mMax = newMax;
                lSum = newL;
            }

            if (lSum > 0f)
                for (int d = 0; d < headDim; d++) pO[d] /= lSum;
            else
                for (int d = 0; d < headDim; d++) pO[d] = 0f;
        }
    }

    // Q8_0 quantized KV cache block constants.
    private const int Q8QK = 32;
    private const int Q8BLOCK = 34; // 2 bytes fp16 scale + 32 × int8

    // Q4_0 quantized KV cache block constants.
    private const int Q4QK = 32;
    private const int Q4BLOCK = 18; // 2 bytes fp16 scale + 16 bytes nibbles

    // Flash Q8_0 quantized KV cache attention kernels.
    // Tiled online softmax O(1) score buffer (stackalloc), no heap alloc.
    

    public static unsafe void ScaledDotProductFlashQ8_0AVX2(
        float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        int nBlocks = (headDim + Q8QK - 1) / Q8QK;
        int colStride = nBlocks * Q8BLOCK;
        float* tileScores = stackalloc float[FlashTileSize];
        int queryBase = causal ? kvLen - seqLen : 0;

        for (int i = 0; i < seqLen; i++)
        {
            float* qi = q + (long)i * qStride;
            int absQPos = queryBase + i;
            int kvOffset = 0;
            int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
            if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

            float mMax = float.NegativeInfinity;
            float lSum = 0f;
            float* pO = output + (long)i * oStride;
            for (int d = 0; d < headDim; d++) pO[d] = 0f;

            for (int start = 0; start < effKvLen; start += FlashTileSize)
            {
                int end = Math.Min(start + FlashTileSize, effKvLen);
                int tileLen = end - start;

                float tileMax = float.NegativeInfinity;
                for (int j = start; j < end; j++)
                {
                    byte* kj_base = kQuant + (long)(kvOffset + j) * colStride;
                    double dot = 0;
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = kj_base + b * Q8BLOCK;
                        float d = HalfToFloat_F16C(*(ushort*)bp);
                        sbyte* vals = (sbyte*)(bp + 2);
                        float* qi_b = qi + b * Q8QK;
                        int blockEnd = Math.Min(Q8QK, headDim - b * Q8QK);

                        var vacc = Vector256<float>.Zero;
                        var vd = Vector256.Create(d);
                        int k = 0;
                        for (; k <= blockEnd - 8; k += 8)
                        {
                            var vi = Vector256.LoadUnsafe(ref qi_b[k]);
                            var vw = Avx.ConvertToVector256Single(
                                Avx2.ConvertToVector256Int32(vals + k));
                            vacc = Avx.Add(vacc,
                                Avx.Multiply(vi, Avx.Multiply(vw, vd)));
                        }
                        float s = MathHelpers.HSum256_Avx(vacc);
                        for (; k < blockEnd; k++)
                            s += qi_b[k] * (vals[k] * d);
                        dot += s;
                    }
                    float score = (float)dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    tileScores[j - start] = score;
                    if (score > tileMax) tileMax = score;
                }

                float newMax = MathF.Max(mMax, tileMax);
                float scaleOld = MathF.Exp(mMax - newMax);
                float scaleNew = 0f;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - newMax);
                    scaleNew += tileScores[t];
                }

                float newL = scaleOld * lSum + scaleNew;
                for (int d = 0; d < headDim; d++)
                    pO[d] *= scaleOld;

                for (int t = 0; t < tileLen; t++)
                {
                    byte* vj_base = vQuant + (long)(kvOffset + start + t) * colStride;
                    float sm = tileScores[t];
                    var vSm = Vector256.Create(sm);
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = vj_base + b * Q8BLOCK;
                        float d = HalfToFloat_F16C(*(ushort*)bp);
                        sbyte* vals = (sbyte*)(bp + 2);
                        int blockEnd = Math.Min(Q8QK, headDim - b * Q8QK);
                        float* pOut = pO + b * Q8QK;
                        var vd = Vector256.Create(d);

                        int k = 0;
                        for (; k <= blockEnd - 8; k += 8)
                        {
                            var vw = Avx.ConvertToVector256Single(
                                Avx2.ConvertToVector256Int32(vals + k));
                            var vOld = Vector256.LoadUnsafe(ref pOut[k]);
                            Vector256.StoreUnsafe(
                                Avx.Add(vOld, Avx.Multiply(vSm,
                                    Avx.Multiply(vw, vd))),
                                ref pOut[k]);
                        }
                        for (; k < blockEnd; k++)
                            pOut[k] += sm * (vals[k] * d);
                    }
                }

                mMax = newMax;
                lSum = newL;
            }

            if (lSum > 0f)
                for (int d = 0; d < headDim; d++) pO[d] /= lSum;
            else
                for (int d = 0; d < headDim; d++) pO[d] = 0f;
        }
    }

    public static unsafe void ScaledDotProductFlashQ8_0FMA(
        float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        int nBlocks = (headDim + Q8QK - 1) / Q8QK;
        int colStride = nBlocks * Q8BLOCK;
        float* tileScores = stackalloc float[FlashTileSize];
        int queryBase = causal ? kvLen - seqLen : 0;

        for (int i = 0; i < seqLen; i++)
        {
            float* qi = q + (long)i * qStride;
            int absQPos = queryBase + i;
            int kvOffset = 0;
            int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
            if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

            float mMax = float.NegativeInfinity;
            float lSum = 0f;
            float* pO = output + (long)i * oStride;
            for (int d = 0; d < headDim; d++) pO[d] = 0f;

            for (int start = 0; start < effKvLen; start += FlashTileSize)
            {
                int end = Math.Min(start + FlashTileSize, effKvLen);
                int tileLen = end - start;

                float tileMax = float.NegativeInfinity;
                for (int j = start; j < end; j++)
                {
                    byte* kj_base = kQuant + (long)(kvOffset + j) * colStride;
                    double dot = 0;
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = kj_base + b * Q8BLOCK;
                        float d = HalfToFloat_F16C(*(ushort*)bp);
                        sbyte* vals = (sbyte*)(bp + 2);
                        float* qi_b = qi + b * Q8QK;
                        int blockEnd = Math.Min(Q8QK, headDim - b * Q8QK);

                        var vacc = Vector256<float>.Zero;
                        var vd = Vector256.Create(d);
                        int k = 0;
                        for (; k <= blockEnd - 8; k += 8)
                        {
                            var vi = Vector256.LoadUnsafe(ref qi_b[k]);
                            var vw = Avx.ConvertToVector256Single(
                                Avx2.ConvertToVector256Int32(vals + k));
                            vacc = Fma.MultiplyAdd(vi,
                                Avx.Multiply(vw, vd), vacc);
                        }
                        float s = MathHelpers.HSum256_Avx(vacc);
                        for (; k < blockEnd; k++)
                            s += qi_b[k] * (vals[k] * d);
                        dot += s;
                    }
                    float score = (float)dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    tileScores[j - start] = score;
                    if (score > tileMax) tileMax = score;
                }

                float newMax = MathF.Max(mMax, tileMax);
                float scaleOld = MathF.Exp(mMax - newMax);
                float scaleNew = 0f;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - newMax);
                    scaleNew += tileScores[t];
                }

                float newL = scaleOld * lSum + scaleNew;
                for (int d = 0; d < headDim; d++)
                    pO[d] *= scaleOld;

                for (int t = 0; t < tileLen; t++)
                {
                    byte* vj_base = vQuant + (long)(kvOffset + start + t) * colStride;
                    float sm = tileScores[t];
                    var vSm = Vector256.Create(sm);
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = vj_base + b * Q8BLOCK;
                        float d = HalfToFloat_F16C(*(ushort*)bp);
                        sbyte* vals = (sbyte*)(bp + 2);
                        int blockEnd = Math.Min(Q8QK, headDim - b * Q8QK);
                        float* pOut = pO + b * Q8QK;
                        var vd = Vector256.Create(d);

                        int k = 0;
                        for (; k <= blockEnd - 8; k += 8)
                        {
                            var vw = Avx.ConvertToVector256Single(
                                Avx2.ConvertToVector256Int32(vals + k));
                            var vOld = Vector256.LoadUnsafe(ref pOut[k]);
                            Vector256.StoreUnsafe(
                                Avx.Add(vOld, Avx.Multiply(vSm,
                                    Avx.Multiply(vw, vd))),
                                ref pOut[k]);
                        }
                        for (; k < blockEnd; k++)
                            pOut[k] += sm * (vals[k] * d);
                    }
                }

                mMax = newMax;
                lSum = newL;
            }

            if (lSum > 0f)
                for (int d = 0; d < headDim; d++) pO[d] /= lSum;
            else
                for (int d = 0; d < headDim; d++) pO[d] = 0f;
        }
    }

    public static unsafe void ScaledDotProductFlashQ8_0Scalar(
        float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        int nBlocks = (headDim + Q8QK - 1) / Q8QK;
        int colStride = nBlocks * Q8BLOCK;
        float* tileScores = stackalloc float[FlashTileSize];
        int queryBase = causal ? kvLen - seqLen : 0;

        for (int i = 0; i < seqLen; i++)
        {
            float* qi = q + (long)i * qStride;
            int absQPos = queryBase + i;
            int kvOffset = 0;
            int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
            if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

            float mMax = float.NegativeInfinity;
            float lSum = 0f;
            float* pO = output + (long)i * oStride;
            for (int d = 0; d < headDim; d++) pO[d] = 0f;

            for (int start = 0; start < effKvLen; start += FlashTileSize)
            {
                int end = Math.Min(start + FlashTileSize, effKvLen);
                int tileLen = end - start;

                float tileMax = float.NegativeInfinity;
                for (int j = start; j < end; j++)
                {
                    byte* kj_base = kQuant + (long)(kvOffset + j) * colStride;
                    double dot = 0;
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = kj_base + b * Q8BLOCK;
                        float d = HalfToFloat_Scalar(*(ushort*)bp);
                        sbyte* vals = (sbyte*)(bp + 2);
                        float* qi_b = qi + b * Q8QK;
                        int blockEnd = Math.Min(Q8QK, headDim - b * Q8QK);
                        for (int k = 0; k < blockEnd; k++)
                            dot += qi_b[k] * (vals[k] * d);
                    }
                    float score = (float)dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    tileScores[j - start] = score;
                    if (score > tileMax) tileMax = score;
                }

                float newMax = MathF.Max(mMax, tileMax);
                float scaleOld = MathF.Exp(mMax - newMax);
                float scaleNew = 0f;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - newMax);
                    scaleNew += tileScores[t];
                }

                float newL = scaleOld * lSum + scaleNew;
                for (int d = 0; d < headDim; d++)
                    pO[d] *= scaleOld;

                for (int t = 0; t < tileLen; t++)
                {
                    byte* vj_base = vQuant + (long)(kvOffset + start + t) * colStride;
                    float sm = tileScores[t];
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = vj_base + b * Q8BLOCK;
                        float d = HalfToFloat_Scalar(*(ushort*)bp);
                        sbyte* vals = (sbyte*)(bp + 2);
                        int blockEnd = Math.Min(Q8QK, headDim - b * Q8QK);
                        float* pOut = pO + b * Q8QK;
                        for (int k = 0; k < blockEnd; k++)
                            pOut[k] += sm * (vals[k] * d);
                    }
                }

                mMax = newMax;
                lSum = newL;
            }

            if (lSum > 0f)
                for (int d = 0; d < headDim; d++) pO[d] /= lSum;
            else
                for (int d = 0; d < headDim; d++) pO[d] = 0f;
        }
    }

    // ── Q4_0 quantized KV cache (flash) ──

    public static unsafe void ScaledDotProductFlashQ4_0AVX2(
        float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        int nBlocks = (headDim + Q4QK - 1) / Q4QK;
        int colStride = nBlocks * Q4BLOCK;
        float* tileScores = stackalloc float[FlashTileSize];
        float* vvBuf = stackalloc float[8];
        int queryBase = causal ? kvLen - seqLen : 0;

        for (int i = 0; i < seqLen; i++)
        {
            float* qi = q + (long)i * qStride;
            int absQPos = queryBase + i;
            int kvOffset = 0;
            int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
            if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

            float mMax = float.NegativeInfinity;
            float lSum = 0f;
            float* pO = output + (long)i * oStride;
            for (int d = 0; d < headDim; d++) pO[d] = 0f;

            for (int start = 0; start < effKvLen; start += FlashTileSize)
            {
                int end = Math.Min(start + FlashTileSize, effKvLen);
                int tileLen = end - start;

                float tileMax = float.NegativeInfinity;
                for (int j = start; j < end; j++)
                {
                    byte* kj_base = kQuant + (long)(kvOffset + j) * colStride;
                    double dot = 0;
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = kj_base + b * Q4BLOCK;
                        float dScale = HalfToFloat_F16C(*(ushort*)bp);
                        byte* qs = bp + 2;
                        float* qi_b = qi + b * Q4QK;
                        int blockEnd = Math.Min(Q4QK, headDim - b * Q4QK);

                        int k = 0;
                        for (; k <= blockEnd - 16; k += 16)
                        {
                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vi0 = Vector256.LoadUnsafe(ref qi_b[k]);
                            var vacc0 = Avx.Multiply(vi0, vw0);

                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + 8 + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vi1 = Vector256.LoadUnsafe(ref qi_b[k + 8]);
                            var vacc1 = Avx.Multiply(vi1, vw1);

                            dot += MathHelpers.HSum256_Avx(vacc0 + vacc1);
                        }
                        for (; k <= blockEnd - 8; k += 8)
                        {
                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vi = Vector256.LoadUnsafe(ref qi_b[k]);
                            dot += MathHelpers.HSum256_Avx(Avx.Multiply(vi, vw));
                        }
                        for (; k < blockEnd; k++)
                        {
                            int nib = ((k & 1) == 0) ? (qs[k / 2] & 0x0F) : (qs[k / 2] >> 4);
                            dot += qi_b[k] * ((nib - 8) * dScale);
                        }
                    }
                    float score = (float)dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    tileScores[j - start] = score;
                    if (score > tileMax) tileMax = score;
                }

                float newMax = MathF.Max(mMax, tileMax);
                float scaleOld = MathF.Exp(mMax - newMax);
                float scaleNew = 0f;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - newMax);
                    scaleNew += tileScores[t];
                }

                float newL = scaleOld * lSum + scaleNew;
                for (int d = 0; d < headDim; d++)
                    pO[d] *= scaleOld;

                for (int t = 0; t < tileLen; t++)
                {
                    byte* vj_base = vQuant + (long)(kvOffset + start + t) * colStride;
                    float sm = tileScores[t];
                    var vSm = Vector256.Create(sm);
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = vj_base + b * Q4BLOCK;
                        float dScale = HalfToFloat_F16C(*(ushort*)bp);
                        byte* qs = bp + 2;
                        int blockEnd = Math.Min(Q4QK, headDim - b * Q4QK);
                        float* pOut = pO + b * Q4QK;

                        int k = 0;
                        for (; k <= blockEnd - 16; k += 16)
                        {
                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vOld0 = Vector256.LoadUnsafe(ref pOut[k]);
                            Vector256.StoreUnsafe(Avx.Add(vOld0, Avx.Multiply(vSm, vw0)), ref pOut[k]);

                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + 8 + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vOld1 = Vector256.LoadUnsafe(ref pOut[k + 8]);
                            Vector256.StoreUnsafe(Avx.Add(vOld1, Avx.Multiply(vSm, vw1)), ref pOut[k + 8]);
                        }
                        for (; k <= blockEnd - 8; k += 8)
                        {
                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vOld = Vector256.LoadUnsafe(ref pOut[k]);
                            Vector256.StoreUnsafe(Avx.Add(vOld, Avx.Multiply(vSm, vw)), ref pOut[k]);
                        }
                        for (; k < blockEnd; k++)
                        {
                            int nib = ((k & 1) == 0) ? (qs[k / 2] & 0x0F) : (qs[k / 2] >> 4);
                            pOut[k] += sm * ((nib - 8) * dScale);
                        }
                    }
                }

                mMax = newMax;
                lSum = newL;
            }

            if (lSum > 0f)
                for (int d = 0; d < headDim; d++) pO[d] /= lSum;
            else
                for (int d = 0; d < headDim; d++) pO[d] = 0f;
        }
    }

    public static unsafe void ScaledDotProductFlashQ4_0FMA(
        float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        int nBlocks = (headDim + Q4QK - 1) / Q4QK;
        int colStride = nBlocks * Q4BLOCK;
        float* tileScores = stackalloc float[FlashTileSize];
        float* vvBuf = stackalloc float[8];
        int queryBase = causal ? kvLen - seqLen : 0;

        for (int i = 0; i < seqLen; i++)
        {
            float* qi = q + (long)i * qStride;
            int absQPos = queryBase + i;
            int kvOffset = 0;
            int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
            if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

            float mMax = float.NegativeInfinity;
            float lSum = 0f;
            float* pO = output + (long)i * oStride;
            for (int d = 0; d < headDim; d++) pO[d] = 0f;

            for (int start = 0; start < effKvLen; start += FlashTileSize)
            {
                int end = Math.Min(start + FlashTileSize, effKvLen);
                int tileLen = end - start;

                float tileMax = float.NegativeInfinity;
                for (int j = start; j < end; j++)
                {
                    byte* kj_base = kQuant + (long)(kvOffset + j) * colStride;
                    double dot = 0;
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = kj_base + b * Q4BLOCK;
                        float dScale = HalfToFloat_F16C(*(ushort*)bp);
                        byte* qs = bp + 2;
                        float* qi_b = qi + b * Q4QK;
                        int blockEnd = Math.Min(Q4QK, headDim - b * Q4QK);

                        int k = 0;
                        for (; k <= blockEnd - 16; k += 16)
                        {
                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vi0 = Vector256.LoadUnsafe(ref qi_b[k]);
                            dot += MathHelpers.HSum256_Avx(Fma.MultiplyAdd(vi0, vw0, Vector256<float>.Zero));

                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + 8 + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vi1 = Vector256.LoadUnsafe(ref qi_b[k + 8]);
                            dot += MathHelpers.HSum256_Avx(Fma.MultiplyAdd(vi1, vw1, Vector256<float>.Zero));
                        }
                        for (; k <= blockEnd - 8; k += 8)
                        {
                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vi = Vector256.LoadUnsafe(ref qi_b[k]);
                            dot += MathHelpers.HSum256_Avx(Fma.MultiplyAdd(vi, vw, Vector256<float>.Zero));
                        }
                        for (; k < blockEnd; k++)
                        {
                            int nib = ((k & 1) == 0) ? (qs[k / 2] & 0x0F) : (qs[k / 2] >> 4);
                            dot += qi_b[k] * ((nib - 8) * dScale);
                        }
                    }
                    float score = (float)dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    tileScores[j - start] = score;
                    if (score > tileMax) tileMax = score;
                }

                float newMax = MathF.Max(mMax, tileMax);
                float scaleOld = MathF.Exp(mMax - newMax);
                float scaleNew = 0f;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - newMax);
                    scaleNew += tileScores[t];
                }

                float newL = scaleOld * lSum + scaleNew;
                for (int d = 0; d < headDim; d++)
                    pO[d] *= scaleOld;

                for (int t = 0; t < tileLen; t++)
                {
                    byte* vj_base = vQuant + (long)(kvOffset + start + t) * colStride;
                    float sm = tileScores[t];
                    var vSm = Vector256.Create(sm);
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = vj_base + b * Q4BLOCK;
                        float dScale = HalfToFloat_F16C(*(ushort*)bp);
                        byte* qs = bp + 2;
                        int blockEnd = Math.Min(Q4QK, headDim - b * Q4QK);
                        float* pOut = pO + b * Q4QK;

                        int k = 0;
                        for (; k <= blockEnd - 16; k += 16)
                        {
                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw0 = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vOld0 = Vector256.LoadUnsafe(ref pOut[k]);
                            Vector256.StoreUnsafe(Fma.MultiplyAdd(vSm, vw0, vOld0), ref pOut[k]);

                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + 8 + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw1 = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vOld1 = Vector256.LoadUnsafe(ref pOut[k + 8]);
                            Vector256.StoreUnsafe(Fma.MultiplyAdd(vSm, vw1, vOld1), ref pOut[k + 8]);
                        }
                        for (; k <= blockEnd - 8; k += 8)
                        {
                            for (int sub = 0; sub < 8; sub++)
                            {
                                int idx = k + sub;
                                int nib = ((idx & 1) == 0) ? (qs[idx / 2] & 0x0F) : (qs[idx / 2] >> 4);
                                vvBuf[sub] = (nib - 8) * dScale;
                            }
                            var vw = Vector256.LoadUnsafe(ref vvBuf[0]);
                            var vOld = Vector256.LoadUnsafe(ref pOut[k]);
                            Vector256.StoreUnsafe(Fma.MultiplyAdd(vSm, vw, vOld), ref pOut[k]);
                        }
                        for (; k < blockEnd; k++)
                        {
                            int nib = ((k & 1) == 0) ? (qs[k / 2] & 0x0F) : (qs[k / 2] >> 4);
                            pOut[k] += sm * ((nib - 8) * dScale);
                        }
                    }
                }

                mMax = newMax;
                lSum = newL;
            }

            if (lSum > 0f)
                for (int d = 0; d < headDim; d++) pO[d] /= lSum;
            else
                for (int d = 0; d < headDim; d++) pO[d] = 0f;
        }
    }

    public static unsafe void ScaledDotProductFlashQ4_0Scalar(
        float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope, int windowSize)
    {
        if ((uint)headDim > FlashMaxHeadDim)
            throw new ArgumentOutOfRangeException(nameof(headDim),
                $"headDim {headDim} exceeds FlashMaxHeadDim {FlashMaxHeadDim}.");

        int nBlocks = (headDim + Q4QK - 1) / Q4QK;
        int colStride = nBlocks * Q4BLOCK;
        float* tileScores = stackalloc float[FlashTileSize];
        int queryBase = causal ? kvLen - seqLen : 0;

        for (int i = 0; i < seqLen; i++)
        {
            float* qi = q + (long)i * qStride;
            int absQPos = queryBase + i;
            int kvOffset = 0;
            int effKvLen = causal ? Math.Min(absQPos + 1, kvLen) : kvLen;
            if (windowSize > 0) { kvOffset = Math.Max(0, absQPos - windowSize + 1); effKvLen -= kvOffset; }

            float mMax = float.NegativeInfinity;
            float lSum = 0f;
            float* pO = output + (long)i * oStride;
            for (int d = 0; d < headDim; d++) pO[d] = 0f;

            for (int start = 0; start < effKvLen; start += FlashTileSize)
            {
                int end = Math.Min(start + FlashTileSize, effKvLen);
                int tileLen = end - start;

                float tileMax = float.NegativeInfinity;
                for (int j = start; j < end; j++)
                {
                    byte* kj_base = kQuant + (long)(kvOffset + j) * colStride;
                    double dot = 0;
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = kj_base + b * Q4BLOCK;
                        float dScale = HalfToFloat_Scalar(*(ushort*)bp);
                        byte* qs = bp + 2;
                        float* qi_b = qi + b * Q4QK;
                        int blockEnd = Math.Min(Q4QK, headDim - b * Q4QK);
                        for (int k = 0; k < blockEnd; k++)
                        {
                            int nib = ((k & 1) == 0) ? (qs[k / 2] & 0x0F) : (qs[k / 2] >> 4);
                            dot += qi_b[k] * ((nib - 8) * dScale);
                        }
                    }
                    float score = (float)dot * scale - alibiSlope * (absQPos - kvOffset - j);
                    tileScores[j - start] = score;
                    if (score > tileMax) tileMax = score;
                }

                float newMax = MathF.Max(mMax, tileMax);
                float scaleOld = MathF.Exp(mMax - newMax);
                float scaleNew = 0f;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - newMax);
                    scaleNew += tileScores[t];
                }

                float newL = scaleOld * lSum + scaleNew;
                for (int d = 0; d < headDim; d++)
                    pO[d] *= scaleOld;

                for (int t = 0; t < tileLen; t++)
                {
                    byte* vj_base = vQuant + (long)(kvOffset + start + t) * colStride;
                    float sm = tileScores[t];
                    for (int b = 0; b < nBlocks; b++)
                    {
                        byte* bp = vj_base + b * Q4BLOCK;
                        float dScale = HalfToFloat_Scalar(*(ushort*)bp);
                        byte* qs = bp + 2;
                        int blockEnd = Math.Min(Q4QK, headDim - b * Q4QK);
                        float* pOut = pO + b * Q4QK;
                        for (int k = 0; k < blockEnd; k++)
                        {
                            int nib = ((k & 1) == 0) ? (qs[k / 2] & 0x0F) : (qs[k / 2] >> 4);
                            pOut[k] += sm * ((nib - 8) * dScale);
                        }
                    }
                }

                mMax = newMax;
                lSum = newL;
            }

            if (lSum > 0f)
                for (int d = 0; d < headDim; d++) pO[d] /= lSum;
            else
                for (int d = 0; d < headDim; d++) pO[d] = 0f;
        }
    }
}
