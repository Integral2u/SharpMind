using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using SharpMind.Model;
using SharpMind.Model.Layers.Attention;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// The int8 attention kernels read the <see cref="Int8KVCache"/> row layouts (values:
/// [scale][headDim x int8]; keys: [one scale per 16 values][headDim x int8]), keep the query row in
/// float, score each key with a float-by-int8 dot scaled per key block
/// (score = (Σ_b sk[b] * dot_b) * scale - ALiBi) and accumulate values from int8. The scalar kernel is
/// pinned to a double-precision reference of that arithmetic and, loosely, to float attention on
/// the original rows; the AVX2 and FMA kernels are pinned to the scalar one across the seams: head
/// sizes off the 32-byte and 8-lane strides, key counts across the 64-key tile, causal and
/// bidirectional, a sliding window, and ALiBi.
/// </summary>
public class Int8AttentionKernelTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (int headDim in new[] { 128, 64, 60, 37, 1 })
            foreach (var (seqLen, kvLen) in new[] { (1, 1), (1, 13), (3, 8), (5, 29), (64, 131) })
                foreach (bool causal in new[] { true, false })
                    foreach (int window in new[] { 0, 6 })
                        foreach (float alibi in new[] { 0f, 0.125f })
                            yield return new object[] { headDim, seqLen, kvLen, causal, window, alibi };
    }

    public enum Tier { Scalar, Avx2, Fma }

    private static bool Supported(Tier tier) => tier switch
    {
        Tier.Avx2 => Avx2.IsSupported,
        Tier.Fma => Avx2.IsSupported && Fma.IsSupported,
        _ => true,
    };

    private static unsafe void Run(Tier tier, float[] q, byte[] k, byte[] v, float[] output, int seqLen, int kvLen, int headDim,
        float scale, bool causal, int qStride, int oStride, float alibi, int window)
    {
        fixed (float* pq = q, po = output)
        fixed (byte* pk = k, pv = v)
        {
            switch (tier)
            {
                case Tier.Scalar: AttentionKernels.ScaledDotProductFlashI8Scalar(pq, pk, pv, po, seqLen, kvLen, headDim, scale, causal, qStride, oStride, alibi, window); break;
                case Tier.Avx2: AttentionKernels.ScaledDotProductFlashI8AVX2(pq, pk, pv, po, seqLen, kvLen, headDim, scale, causal, qStride, oStride, alibi, window); break;
                case Tier.Fma: AttentionKernels.ScaledDotProductFlashI8FMA(pq, pk, pv, po, seqLen, kvLen, headDim, scale, causal, qStride, oStride, alibi, window); break;
            }
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Scalar_MatchesTheInt8Reference(int headDim, int seqLen, int kvLen, bool causal, int window, float alibi)
    {
        var (q, k, v, scale) = Inputs(headDim, seqLen, kvLen, 3f);
        var expected = Reference(q, k, v, seqLen, kvLen, headDim, scale, causal, alibi, window);
        var actual = new float[seqLen * headDim];
        Array.Fill(actual, float.NaN);
        Run(Tier.Scalar, q, k, v, actual, seqLen, kvLen, headDim, scale, causal, headDim, headDim, alibi, window);
        AssertClose(expected, actual, $"hd={headDim} S={seqLen} KV={kvLen} causal={causal} win={window} alibi={alibi}", "reference", "scalar");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Avx2_MatchesScalar(int headDim, int seqLen, int kvLen, bool causal, int window, float alibi) =>
        MatchesScalar(Tier.Avx2, headDim, seqLen, kvLen, causal, window, alibi);

    [Theory]
    [MemberData(nameof(Cases))]
    public void Fma_MatchesScalar(int headDim, int seqLen, int kvLen, bool causal, int window, float alibi) =>
        MatchesScalar(Tier.Fma, headDim, seqLen, kvLen, causal, window, alibi);

    private static void MatchesScalar(Tier tier, int headDim, int seqLen, int kvLen, bool causal, int window, float alibi)
    {
        if (!Supported(tier)) return;
        var (q, k, v, scale) = Inputs(headDim, seqLen, kvLen, 3f);
        var expected = new float[seqLen * headDim];
        var actual = new float[seqLen * headDim];
        Array.Fill(actual, float.NaN);
        Run(Tier.Scalar, q, k, v, expected, seqLen, kvLen, headDim, scale, causal, headDim, headDim, alibi, window);
        Run(tier, q, k, v, actual, seqLen, kvLen, headDim, scale, causal, headDim, headDim, alibi, window);
        AssertClose(expected, actual, $"hd={headDim} S={seqLen} KV={kvLen} causal={causal} win={window} alibi={alibi}", "scalar", tier.ToString());
    }

    /// <summary>
    /// Guards the reference itself: the same wrong scale formula in kernel and reference would still
    /// agree, but not with float attention on the rows before quantization.
    /// </summary>
    [Theory]
    [InlineData(128, 1, 577)]
    [InlineData(64, 19, 19)]
    [InlineData(37, 5, 29)]
    public unsafe void Scalar_IsCloseToFloatAttentionOnTheOriginalRows(int headDim, int seqLen, int kvLen)
    {
        var rng = new Random(77 + headDim + kvLen);
        var q = RandomArray(seqLen * headDim, rng, 2f);
        var kf = RandomArray(kvLen * headDim, rng, 2f);
        var vf = RandomArray(kvLen * headDim, rng, 1f);
        float scale = 1f / MathF.Sqrt(headDim);
        var floatOut = new float[seqLen * headDim];
        fixed (float* pq = q, pk = kf, pv = vf, po = floatOut)
            AttentionKernels.ScaledDotProductFlashScalar(pq, pk, pv, po, seqLen, kvLen, headDim, scale, true, headDim, headDim, 0f, 0);

        var int8Out = new float[seqLen * headDim];
        Run(Tier.Scalar, q, QuantizeKeyRows(kf, headDim), QuantizeRows(vf, headDim), int8Out, seqLen, kvLen, headDim, scale, true, headDim, headDim, 0f, 0);

        for (int i = 0; i < seqLen; i++)
        {
            var a = floatOut.AsSpan(i * headDim, headDim);
            var b = int8Out.AsSpan(i * headDim, headDim);
            double dot = 0, na = 0, nb = 0, nd = 0;
            for (int d = 0; d < headDim; d++) { dot += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d]; nd += (a[d] - b[d]) * (a[d] - b[d]); }
            double cosine = dot / Math.Sqrt(na * nb);
            double relError = Math.Sqrt(nd / na);
            Assert.True(cosine >= 0.999 && relError <= 0.05, $"hd={headDim} S={seqLen} KV={kvLen} query {i}: cosine {cosine:F5}, relative error {relError:F4}");
        }
    }

    /// <summary>
    /// Production calls step queries and outputs by the whole layer's width, not one head's. Padded
    /// strides catch a kernel that steps by headDim instead; NaN everywhere outside the head's own
    /// output slots catches writes into a neighbouring head.
    /// </summary>
    [Theory]
    [InlineData(Tier.Scalar)]
    [InlineData(Tier.Avx2)]
    [InlineData(Tier.Fma)]
    public void HonoursQueryAndOutputStridesAndWritesOnlyItsSlots(Tier tier)
    {
        if (!Supported(tier)) return;
        const int headDim = 60, seqLen = 7, kvLen = 21, qStride = headDim * 5 + 3, oStride = headDim * 3 + 11;
        var rng = new Random(2024);
        var q = RandomArray(seqLen * qStride, rng, 3f);
        var k = QuantizeKeyRows(RandomArray(kvLen * headDim, rng, 3f), headDim);
        var v = QuantizeRows(RandomArray(kvLen * headDim, rng, 1f), headDim);
        float scale = 1f / MathF.Sqrt(headDim);

        var compact = new float[seqLen * headDim];
        for (int i = 0; i < seqLen; i++) Array.Copy(q, i * qStride, compact, i * headDim, headDim);
        var expected = Reference(compact, k, v, seqLen, kvLen, headDim, scale, true, 0f, 0);

        var actual = new float[seqLen * oStride];
        Array.Fill(actual, float.NaN);
        Run(tier, q, k, v, actual, seqLen, kvLen, headDim, scale, true, qStride, oStride, 0f, 0);

        for (int i = 0; i < actual.Length; i++)
        {
            if (i % oStride < headDim)
            {
                float e = expected[i / oStride * headDim + i % oStride];
                Assert.True(Math.Abs(e - actual[i]) <= 2e-5f + 2e-5f * Math.Abs(e), $"{tier} [{i / oStride},{i % oStride}]: reference={e} kernel={actual[i]}");
            }
            else
                Assert.True(float.IsNaN(actual[i]), $"{tier}: slot {i} outside the head's output was written: {actual[i]}");
        }
    }

    /// <summary>Rows with no visible key (a causal chunk longer than the cache) and rows whose scores are NaN are zero.</summary>
    [Theory]
    [InlineData(Tier.Scalar)]
    [InlineData(Tier.Avx2)]
    [InlineData(Tier.Fma)]
    public void RowsWithoutVisibleKeysOrWithNaNScoresAreZero(Tier tier)
    {
        if (!Supported(tier)) return;
        const int headDim = 16, seqLen = 5, kvLen = 2;
        var rng = new Random(8);
        var q = RandomArray(seqLen * headDim, rng, 1f);
        var k = QuantizeKeyRows(RandomArray(kvLen * headDim, rng, 1f), headDim);
        var v = QuantizeRows(RandomArray(kvLen * headDim, rng, 1f), headDim);
        for (int d = 0; d < headDim; d++) q[4 * headDim + d] = float.NaN;   // last query: NaN scores
        var actual = new float[seqLen * headDim];
        Array.Fill(actual, float.NaN);
        Run(tier, q, k, v, actual, seqLen, kvLen, headDim, 0.25f, true, headDim, headDim, 0f, 0);

        // queryBase = kvLen - seqLen = -3: queries 0-2 see no key; 3 sees both; 4 has NaN scores.
        for (int i = 0; i < 3 * headDim; i++) Assert.Equal(0f, actual[i]);
        var expected = Reference(q, k, v, seqLen, kvLen, headDim, 0.25f, true, 0f, 0);
        for (int i = 3 * headDim; i < 4 * headDim; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-5f + 2e-5f * Math.Abs(expected[i]), $"{tier} [3,{i % headDim}]: reference={expected[i]} kernel={actual[i]}");
        for (int i = 4 * headDim; i < 5 * headDim; i++) Assert.Equal(0f, actual[i]);
    }

    // ── helpers ──

    private static (float[] Q, byte[] K, byte[] V, float Scale) Inputs(int headDim, int seqLen, int kvLen, float range)
    {
        var rng = new Random(1234 + headDim + seqLen * 31 + kvLen * 7);
        var q = RandomArray(seqLen * headDim, rng, range);
        var k = QuantizeKeyRows(RandomArray(kvLen * headDim, rng, range), headDim);
        var v = QuantizeRows(RandomArray(kvLen * headDim, rng, 1f), headDim);
        return (q, k, v, 1f / MathF.Sqrt(headDim));
    }

    // The layout is spelled out here rather than read from Int8KVCache, so a kernel or cache that
    // drifts from it (block size, scales-first order, either stride) disagrees with the reference.
    private const int KeyBlock = 16;
    private static int KeyBlocks(int headDim) => (headDim + KeyBlock - 1) / KeyBlock;
    private static int KeyRowBytes(int headDim) => KeyBlocks(headDim) * sizeof(float) + headDim;

    private static unsafe byte[] QuantizeKeyRows(float[] rows, int headDim)
    {
        int count = rows.Length / headDim, rowBytes = KeyRowBytes(headDim);
        var bytes = new byte[count * rowBytes];
        fixed (float* src = rows)
        fixed (byte* dst = bytes)
            for (int r = 0; r < count; r++)
                Int8KVCache.QuantizeKeyRow(src + (long)r * headDim, dst + (long)r * rowBytes, headDim);
        return bytes;
    }

    private static unsafe byte[] QuantizeRows(float[] rows, int headDim)
    {
        int count = rows.Length / headDim, rowBytes = sizeof(float) + headDim;
        var bytes = new byte[count * rowBytes];
        fixed (float* src = rows)
        fixed (byte* dst = bytes)
            for (int r = 0; r < count; r++)
                Int8KVCache.QuantizeRow(src + (long)r * headDim, dst + (long)r * rowBytes, headDim);
        return bytes;
    }

    /// <summary>The int8 attention arithmetic in double precision, one query row at a time.</summary>
    private static unsafe float[] Reference(float[] q, byte[] k, byte[] v, int seqLen, int kvLen, int headDim,
        float scale, bool causal, float alibi, int window)
    {
        int valueRowBytes = sizeof(float) + headDim, keyRowBytes = KeyRowBytes(headDim), keyData = KeyBlocks(headDim) * sizeof(float);
        var output = new float[seqLen * headDim];
        int queryBase = causal ? kvLen - seqLen : 0;
        for (int i = 0; i < seqLen; i++)
        {
            int absQ = queryBase + i;
            int kvOffset = 0;
            int visible = causal ? Math.Min(absQ + 1, kvLen) : kvLen;
            if (window > 0) { kvOffset = Math.Max(0, absQ - window + 1); visible -= kvOffset; }
            if (visible <= 0) continue;

            var scores = new double[visible];
            for (int j = 0; j < visible; j++)
            {
                int row = (kvOffset + j) * keyRowBytes;
                double dot = 0;
                for (int d = 0; d < headDim; d++)
                    dot += (double)q[i * headDim + d] * (sbyte)k[row + keyData + d] * BitConverter.ToSingle(k, row + d / KeyBlock * sizeof(float));
                scores[j] = dot * scale - alibi * (absQ - kvOffset - j);
            }
            double max = scores.Max();
            if (double.IsNaN(max) || scores.Any(double.IsNaN)) continue;
            double sum = scores.Sum(s => Math.Exp(s - max));
            for (int d = 0; d < headDim; d++)
            {
                double acc = 0;
                for (int j = 0; j < visible; j++)
                {
                    int row = (kvOffset + j) * valueRowBytes;
                    acc += Math.Exp(scores[j] - max) / sum * BitConverter.ToSingle(v, row) * (sbyte)v[row + sizeof(float) + d];
                }
                output[i * headDim + d] = (float)acc;
            }
        }
        return output;
    }

    private static void AssertClose(float[] expected, float[] actual, string label, string expectedName, string actualName)
    {
        int headDim = label.Contains("hd=") ? int.Parse(label[(label.IndexOf("hd=") + 3)..].Split(' ')[0]) : 1;
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-5f + 2e-5f * Math.Abs(expected[i]),
                $"{label} [{i / headDim},{i % headDim}]: {expectedName}={expected[i]} {actualName}={actual[i]}");
    }

    private static float[] RandomArray(int n, Random rng, float range)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * range;
        return x;
    }
}
