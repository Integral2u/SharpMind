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
/// <para><b>Why two forms.</b> Expanding K/V costs <c>rows × kvLora ×
/// heads(nope+v)</c>; absorbing costs <c>rows × kvLora</c> for the scores plus
/// <c>kvLora × vDim</c> to read the values back out, per head. Prefill has many
/// query rows sharing one expansion, so expanding wins; decode has one query row
/// over the whole cache, so absorbing wins by an order of magnitude (≈2M
/// multiply-adds per layer per token against ≈65k once the query is absorbed).
/// llama.cpp makes the same split.</para>
/// </summary>
internal static unsafe class MlaAttention
{
    /// <summary>
    /// Absorbed decode for one query row. <paramref name="qAbs"/> is scratch of
    /// <paramref name="kvLora"/> floats, <paramref name="probs"/> scratch of
    /// <paramref name="len"/>, <paramref name="outLatent"/> scratch of
    /// <paramref name="kvLora"/>, and <paramref name="output"/> is
    /// <paramref name="heads"/> × <paramref name="vDim"/>.
    ///
    /// <para><paramref name="w"/> is row-major <c>[kvLora, heads × (nope +
    /// vDim)]</c> — the loader transposes to [in, out], so column
    /// <c>h*(nope+vDim) + j</c> is head h's j-th output.</para>
    /// </summary>
    public static void Decode(
        float* qNope, float* qRope, float* qAbs,
        float* latents, int len, float* w,
        int kvLora, int heads, int nope, int vDim, int ropeDim, float scale,
        float* probs, float* outLatent, float* output)
    {
        int headStride = nope + vDim;
        int outStride = heads * headStride;
        int latentStride = kvLora + ropeDim;

        for (int h = 0; h < heads; h++)
        {
            float* qn = qNope + (long)h * nope;
            float* qr = qRope + (long)h * ropeDim;

            // q_abs = W_k_b[h]ᵀ · q_nope: the key projection folded into the
            // query, so a score against a cached latent is one kvLora-dot.
            for (int j = 0; j < kvLora; j++)
            {
                float* wRow = w + (long)j * outStride + h * headStride;
                float sum = 0;
                for (int i = 0; i < nope; i++) sum += qn[i] * wRow[i];
                qAbs[j] = sum;
            }

            for (int t = 0; t < len; t++)
            {
                float* row = latents + (long)t * latentStride;
                float s = 0;
                for (int j = 0; j < kvLora; j++) s += qAbs[j] * row[j];
                for (int r = 0; r < ropeDim; r++) s += qr[r] * row[kvLora + r];
                probs[t] = s * scale;
            }

            Softmax(probs, len);

            // Value readout in latent space, then W_v_b[h] back to vDim.
            for (int j = 0; j < kvLora; j++) outLatent[j] = 0;
            for (int t = 0; t < len; t++)
            {
                float p = probs[t];
                if (p == 0f) continue;
                float* row = latents + (long)t * latentStride;
                for (int j = 0; j < kvLora; j++) outLatent[j] += p * row[j];
            }

            float* outHead = output + (long)h * vDim;
            for (int r = 0; r < vDim; r++)
            {
                float sum = 0;
                for (int j = 0; j < kvLora; j++)
                    sum += outLatent[j] * w[(long)j * outStride + h * headStride + nope + r];
                outHead[r] = sum;
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
                    float sum = 0;
                    for (int i = 0; i < nope; i++) sum += qn[i] * k[i];
                    for (int r = 0; r < ropeDim; r++) sum += qr[r] * lat[kvLora + r];
                    probs[t] = sum * scale;
                }

                Softmax(probs, q + 1);

                float* outHead = output + ((long)s * heads + h) * vDim;
                for (int v = 0; v < vDim; v++)
                {
                    float sum = 0;
                    for (int t = 0; t <= q; t++)
                    {
                        float p = probs[t];
                        if (p == 0f) continue;
                        sum += p * kv[(long)t * kvStride + h * headStride + nope + v];
                    }
                    outHead[v] = sum;
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
    /// <paramref name="dst"/> <c>[rows, heads × (nope + vDim)]</c> — the same
    /// product <c>Wv.Forward</c> computes, over a raw latent buffer.
    /// </summary>
    public static void Expand(
        float* latents, int rows, float* w, int kvLora,
        int heads, int nope, int vDim, int ropeDim, float* dst)
    {
        int outStride = heads * (nope + vDim);
        int latentStride = kvLora + ropeDim;
        for (int t = 0; t < rows; t++)
        {
            float* lat = latents + (long)t * latentStride;
            float* outRow = dst + (long)t * outStride;
            for (int o = 0; o < outStride; o++)
            {
                float sum = 0;
                for (int j = 0; j < kvLora; j++) sum += lat[j] * w[(long)j * outStride + o];
                outRow[o] = sum;
            }
        }
    }
}
