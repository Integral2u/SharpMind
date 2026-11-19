using System.Numerics;
using System.Runtime.CompilerServices;

namespace SharpMind.Model.Layers.Attention;

/// <summary>
/// The two attention forms Multi-head Latent Attention needs, over ABIs that are
/// just pointers and counts so they can be tested without a model.
///
/// <para><b>Geometry</b> (DeepSeek V2/V3, Moonlight): a head is
/// <c>nope + rope = 192</c> wide, values are <c>vDim = 128</c> wide, the cached
/// latent is <c>kvLora + rope = 576</c> wide, and <c>kv_b</c> maps the 512-wide
/// latent to <c>heads × (nope ‖ v)</c> — one fused matrix holding both the key
/// and the value of every head.</para>
///
/// <para><b>Layout.</b> <paramref name="w"/> is stored head-major:
/// <c>w[(h * kvLora + j) * headStride + r]</c>, where
/// <c>headStride = nope + vDim</c>. Every kernel here walks one head at a time,
/// and this layout makes each of those walks sequential; see
/// <see cref="MlaKvBLayout"/> for the packing and the measured difference.</para>
///
/// <para><b>Why two forms.</b> Expanding K/V costs <c>rows × kvLora ×
/// heads(nope+v)</c>; absorbing costs <c>rows × kvLora</c> for the scores plus
/// <c>kvLora × vDim</c> to read the values back out, per head. Prefill has many
/// query rows sharing one expansion, so expanding wins; decode has one query row
/// over the whole cache, so absorbing wins by an order of magnitude (≈2M
/// multiply-adds per layer per token against ≈65k). llama.cpp makes the same
/// split.</para>
///
/// <para><b>Why hand-vectorised.</b> These loops are dot products and scaled
/// adds over contiguous floats. The C# JIT will not vectorise a floating-point
/// reduction (it may not reorder the adds), so the naturally written
/// <c>sum += a[i] * b[i]</c> runs on one FMA dependency chain: 0.22 GMAC/s here
/// against 0.56 with <see cref="Vector{T}"/>, and 1.29 once the matrix is also
/// packed head-major. <see cref="Vector{T}"/> is 8-wide on AVX2 and 4-wide on
/// Arm64 NEON from the same source, so no per-tier variants are needed.</para>
/// </summary>
internal static unsafe class MlaAttention
{
    /// <summary>
    /// Absorbed decode for one query row. <paramref name="qAbs"/> is scratch of
    /// <paramref name="kvLora"/> floats, <paramref name="probs"/> scratch of
    /// <paramref name="len"/>, <paramref name="outLatent"/> scratch of
    /// <paramref name="kvLora"/>, and <paramref name="output"/> is
    /// <paramref name="heads"/> × <paramref name="vDim"/>.
    /// </summary>
    public static void Decode(
        float* qNope, float* qRope, float* qAbs,
        float* latents, int len, float* w,
        int kvLora, int heads, int nope, int vDim, int ropeDim, float scale,
        float* probs, float* outLatent, float* output)
    {
        int headStride = nope + vDim;
        int latentStride = kvLora + ropeDim;

        for (int h = 0; h < heads; h++)
        {
            float* qn = qNope + (long)h * nope;
            float* qr = qRope + (long)h * ropeDim;
            float* wh = w + (long)h * kvLora * headStride;

            // q_abs = W_k_b[h]ᵀ · q_nope: the key projection folded into the
            // query, so a score against a cached latent is one kvLora-dot.
            for (int j = 0; j < kvLora; j++)
                qAbs[j] = Dot(qn, wh + (long)j * headStride, nope);

            for (int t = 0; t < len; t++)
            {
                float* row = latents + (long)t * latentStride;
                float s = Dot(qAbs, row, kvLora) + Dot(qr, row + kvLora, ropeDim);
                probs[t] = s * scale;
            }

            Softmax(probs, len);

            // Value readout in latent space, then W_v_b[h] back to vDim.
            for (int j = 0; j < kvLora; j++) outLatent[j] = 0;
            for (int t = 0; t < len; t++)
            {
                float p = probs[t];
                if (p == 0f) continue;
                Saxpy(outLatent, latents + (long)t * latentStride, p, kvLora);
            }

            float* outHead = output + (long)h * vDim;
            for (int r = 0; r < vDim; r++) outHead[r] = 0;
            for (int j = 0; j < kvLora; j++)
            {
                float l = outLatent[j];
                if (l == 0f) continue;
                Saxpy(outHead, wh + (long)j * headStride + nope, l, vDim);
            }
        }
    }

    /// <summary>
    /// Causal prefill over expanded K/V. <paramref name="kv"/> is
    /// <c>[rows, heads × (nope + vDim)]</c> from <see cref="Expand"/>; the
    /// queries are the last <paramref name="seqLen"/> rows, so query row s sits
    /// at position <c>rows - seqLen + s</c> and attends up to and including it.
    /// </summary>
    public static void Prefill(
        float* qNope, float* qRope, int seqLen,
        float* latents, float* kv, int rows,
        int kvLora, int heads, int nope, int vDim, int ropeDim, float scale,
        float* probs, float* output)
    {
        int headStride = nope + vDim;
        int kvStride = heads * headStride;
        int latentStride = kvLora + ropeDim;
        int firstQuery = rows - seqLen;

        for (int h = 0; h < heads; h++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int q = firstQuery + s;
                float* qn = qNope + ((long)s * heads + h) * nope;
                float* qr = qRope + ((long)s * heads + h) * ropeDim;

                for (int t = 0; t <= q; t++)
                {
                    float* k = kv + (long)t * kvStride + h * headStride;
                    float* lat = latents + (long)t * latentStride;
                    float sum = Dot(qn, k, nope) + Dot(qr, lat + kvLora, ropeDim);
                    probs[t] = sum * scale;
                }

                Softmax(probs, q + 1);

                float* outHead = output + ((long)s * heads + h) * vDim;
                for (int v = 0; v < vDim; v++) outHead[v] = 0;
                for (int t = 0; t <= q; t++)
                {
                    float p = probs[t];
                    if (p == 0f) continue;
                    Saxpy(outHead, kv + (long)t * kvStride + h * headStride + nope, p, vDim);
                }
            }
        }
    }

    /// <summary>Numerically stable softmax over the first <paramref name="n"/> entries.</summary>
    public static void Softmax(float* values, int n)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
            if (values[i] > max) max = values[i];

        float sum = 0;
        for (int i = 0; i < n; i++)
        {
            values[i] = MathF.Exp(values[i] - max);
            sum += values[i];
        }
        if (sum > 0)
        {
            float inv = 1f / sum;
            for (int i = 0; i < n; i++) values[i] *= inv;
        }
    }

    /// <summary>
    /// Expands <paramref name="rows"/> latents through kv_b into
    /// <paramref name="dst"/> <c>[rows, heads × (nope + vDim)]</c> — one scaled
    /// add over a head's contiguous 256-wide row per latent element, so each
    /// input row is streamed exactly once.
    /// </summary>
    public static void Expand(
        float* latents, int rows, float* w, int kvLora,
        int heads, int nope, int vDim, int ropeDim, float* dst)
    {
        int headStride = nope + vDim;
        int latentStride = kvLora + ropeDim;
        int outStride = heads * headStride;

        for (int t = 0; t < rows; t++)
        {
            float* lat = latents + (long)t * latentStride;
            float* outRow = dst + (long)t * outStride;
            for (int h = 0; h < heads; h++)
            {
                float* o = outRow + (long)h * headStride;
                float* wh = w + (long)h * kvLora * headStride;
                for (int r = 0; r < headStride; r++) o[r] = 0;
                for (int j = 0; j < kvLora; j++)
                {
                    float l = lat[j];
                    if (l == 0f) continue;
                    Saxpy(o, wh + (long)j * headStride, l, headStride);
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Load(float* p, int width) => new(new ReadOnlySpan<float>(p, width));

    /// <summary>Dot product over contiguous floats, two vector accumulators.</summary>
    internal static float Dot(float* a, float* b, int n)
    {
        int width = Vector<float>.Count;
        var acc0 = Vector<float>.Zero;
        var acc1 = Vector<float>.Zero;
        int i = 0;
        for (; i + 2 * width <= n; i += 2 * width)
        {
            acc0 += Load(a + i, width) * Load(b + i, width);
            acc1 += Load(a + i + width, width) * Load(b + i + width, width);
        }
        for (; i + width <= n; i += width)
            acc0 += Load(a + i, width) * Load(b + i, width);

        var total = acc0 + acc1;
        float sum = 0f;
        for (int k = 0; k < width; k++) sum += total[k];
        for (; i < n; i++) sum += a[i] * b[i];
        return sum;
    }

    /// <summary><c>dst += s * src</c> over contiguous floats.</summary>
    internal static void Saxpy(float* dst, float* src, float s, int n)
    {
        int width = Vector<float>.Count;
        var sv = new Vector<float>(s);
        int i = 0;
        for (; i + width <= n; i += width)
        {
            var v = Load(dst + i, width) + sv * Load(src + i, width);
            v.CopyTo(new Span<float>(dst + i, width));
        }
        for (; i < n; i++) dst[i] += s * src[i];
    }
}
