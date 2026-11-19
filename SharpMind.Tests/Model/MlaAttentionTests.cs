using System;
using SharpMind.Model.Layers.Attention;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// MLA is two different algorithms over the same numbers: prefill expands the
/// latent through kv_b, decode absorbs kv_b into the query. Both must produce
/// the same attention output for the same row, which is a real cross-check
/// because neither path calls the other. The naive reference below is index
/// arithmetic (no pointers), so stride mistakes in the kernels show up as a
/// mismatch instead of agreeing with themselves.
/// </summary>
public unsafe class MlaAttentionTests
{
    private const int Heads = 2;
    private const int Nope = 3;
    private const int Rope = 2;
    private const int VDim = 2;
    private const int KvLora = 4;
    private const int Rows = 3;
    private const int HeadStride = Nope + VDim;

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    /// <summary>
    /// Reorders the plain <c>[kvLora, heads * headStride]</c> matrix the reference
    /// indexes into the head-major blocks the kernels read.
    /// </summary>
    private static float[] Packed(float[] plain)
    {
        int outStride = Heads * HeadStride;
        var packed = new float[plain.Length];
        for (int h = 0; h < Heads; h++)
            for (int j = 0; j < KvLora; j++)
                for (int r = 0; r < HeadStride; r++)
                    packed[(h * KvLora + j) * HeadStride + r] = plain[j * outStride + h * HeadStride + r];
        return packed;
    }

    /// <summary>
    /// One query row against <see cref="Rows"/> cached latents, computed with
    /// plain index arithmetic and a hand-written softmax.
    /// </summary>
    private static float[] ReferenceDecode(
        float[] qNope, float[] qRope, float[] latents, float[] w, float scale)
    {
        var qAbs = new float[KvLora];
        var output = new float[Heads * VDim];
        int headStride = Nope + VDim;
        int outStride = Heads * headStride;
        int latentStride = KvLora + Rope;

        for (int h = 0; h < Heads; h++)
        {
            for (int j = 0; j < KvLora; j++)
            {
                double sum = 0;
                for (int i = 0; i < Nope; i++)
                    sum += qNope[h * Nope + i] * w[j * outStride + h * headStride + i];
                qAbs[j] = (float)sum;
            }

            var probs = new double[Rows];
            double max = double.NegativeInfinity;
            for (int t = 0; t < Rows; t++)
            {
                double s = 0;
                for (int j = 0; j < KvLora; j++)
                    s += qAbs[j] * latents[t * latentStride + j];
                for (int r = 0; r < Rope; r++)
                    s += qRope[h * Rope + r] * latents[t * latentStride + KvLora + r];
                probs[t] = s * scale;
                if (probs[t] > max) max = probs[t];
            }

            double denom = 0;
            for (int t = 0; t < Rows; t++)
            {
                probs[t] = Math.Exp(probs[t] - max);
                denom += probs[t];
            }
            for (int t = 0; t < Rows; t++) probs[t] /= denom;

            var outLatent = new double[KvLora];
            for (int t = 0; t < Rows; t++)
                for (int j = 0; j < KvLora; j++)
                    outLatent[j] += probs[t] * latents[t * latentStride + j];

            for (int r = 0; r < VDim; r++)
            {
                double sum = 0;
                for (int j = 0; j < KvLora; j++)
                    sum += outLatent[j] * w[j * outStride + h * headStride + Nope + r];
                output[h * VDim + r] = (float)sum;
            }
        }
        return output;
    }

    [Fact]
    public void Decode_MatchesNaiveReference()
    {
        var qNope = Random(Heads * Nope, 11);
        var qRope = Random(Heads * Rope, 12);
        var latents = Random(Rows * (KvLora + Rope), 13);
        var w = Random(KvLora * Heads * (Nope + VDim), 14);
        const float scale = 0.5f;

        var expected = ReferenceDecode(qNope, qRope, latents, w, scale);

        var packed = Packed(w);
        var qAbs = new float[KvLora];
        var probs = new float[Rows];
        var outLatent = new float[KvLora];
        var actual = new float[Heads * VDim];

        fixed (float* pqN = qNope, pqR = qRope, pl = latents, pw = packed,
               pa = qAbs, pp = probs, po = outLatent, pOut = actual)
        {
            MlaAttention.Decode(pqN, pqR, pa, pl, Rows, pw,
                KvLora, Heads, Nope, VDim, Rope, scale, pp, po, pOut);
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], actual[i], 5);
    }

    [Fact]
    public void Decode_AndPrefill_AgreeOnTheSameRow()
    {
        var qNope = Random(Heads * Nope, 21);
        var qRope = Random(Heads * Rope, 22);
        var latents = Random(Rows * (KvLora + Rope), 23);
        var w = Random(KvLora * Heads * (Nope + VDim), 24);
        const float scale = 0.7f;

        var packed = Packed(w);
        var decoded = new float[Heads * VDim];
        var qAbs = new float[KvLora];
        var probs = new float[Rows];
        var outLatent = new float[KvLora];
        var kv = new float[Rows * Heads * (Nope + VDim)];
        var prefilled = new float[Heads * VDim];

        fixed (float* pqN = qNope, pqR = qRope, pl = latents, pw = packed,
               pa = qAbs, pp = probs, po = outLatent, pkv = kv,
               pD = decoded, pP = prefilled)
        {
            MlaAttention.Decode(pqN, pqR, pa, pl, Rows, pw,
                KvLora, Heads, Nope, VDim, Rope, scale, pp, po, pD);

            MlaAttention.Expand(pl, Rows, pw, KvLora, Heads, Nope, VDim, Rope, pkv);

            // One query row, at the last cached position: causal attention over
            // every row. The kernels have nothing else in common.
            MlaAttention.Prefill(pqN, pqR, 1, pl, pkv, Rows,
                KvLora, Heads, Nope, VDim, Rope, scale, pp, pP);
        }

        for (int i = 0; i < decoded.Length; i++)
            Assert.Equal(decoded[i], prefilled[i], 5);
    }

    [Fact]
    public void Softmax_SumsToOne_AndToleratesLargeLogits()
    {
        var values = new[] { 1000f, 1001f, 999f, -1000f };
        fixed (float* p = values)
            MlaAttention.Softmax(p, values.Length);

        double sum = 0;
        foreach (float v in values) sum += v;
        Assert.Equal(1.0, sum, 6);
        Assert.True(values[1] > values[0] && values[0] > values[2]);
        Assert.Equal(0f, values[3], 6);
    }

    /// <summary>
    /// The loader hands kv_b over in GGUF order (<c>[out][in]</c>); the kernels
    /// read head-major blocks. A mistake here still produces plausible numbers,
    /// so the permutation is pinned element by element.
    /// </summary>
    [Fact]
    public void Pack_MovesGgufOrderIntoHeadMajorBlocks()
    {
        var src = Random(KvLora * Heads * HeadStride, 31);
        var dst = new float[src.Length];

        MlaKvBLayout.Pack(src, KvLora, Heads, HeadStride, dst);

        for (int h = 0; h < Heads; h++)
            for (int j = 0; j < KvLora; j++)
                for (int r = 0; r < HeadStride; r++)
                    Assert.Equal(src[(h * HeadStride + r) * KvLora + j], dst[(h * KvLora + j) * HeadStride + r]);
    }

    /// <summary>Packed and plain data must describe the same matrix.</summary>
    [Fact]
    public void Pack_AgreesWithTheReferencePermutation()
    {
        var plain = Random(KvLora * Heads * HeadStride, 32);

        // Pack() takes GGUF order, so feed it the transpose of the plain matrix.
        var ggufOrder = new float[plain.Length];
        for (int j = 0; j < KvLora; j++)
            for (int o = 0; o < Heads * HeadStride; o++)
                ggufOrder[o * KvLora + j] = plain[j * Heads * HeadStride + o];

        var packed = new float[plain.Length];
        MlaKvBLayout.Pack(ggufOrder, KvLora, Heads, HeadStride, packed);

        Assert.Equal(Packed(plain), packed);
    }
}
