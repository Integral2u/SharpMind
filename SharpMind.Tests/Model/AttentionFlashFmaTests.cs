using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics.X86;
using SharpMind.Model.Layers.Attention;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// The FMA attention kernel scores eight keys per pass, softmaxes the whole row with
/// Vector256.Exp and accumulates values four rows per pass. Its only other tests compare it with
/// itself (serial vs parallel head loops), so this pins its numbers to the scalar flash kernel
/// across the seams: key counts off the 8- and 4-row tiles, a single key, a head size off the
/// 8-lane stride, causal and bidirectional, a sliding window, and ALiBi.
/// </summary>
public class AttentionFlashFmaTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (int headDim in new[] { 64, 60, 9, 7, 1 })
            foreach (var (seqLen, kvLen) in new[] { (1, 1), (1, 7), (1, 13), (3, 8), (5, 29), (64, 64), (64, 131) })
                foreach (bool causal in new[] { true, false })
                    foreach (int window in new[] { 0, 6 })
                        foreach (float alibi in new[] { 0f, 0.125f })
                            yield return new object[] { headDim, seqLen, kvLen, causal, window, alibi };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public unsafe void MatchesScalarFlashKernel(int headDim, int seqLen, int kvLen, bool causal, int window, float alibi)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        var rng = new Random(1234 + headDim + seqLen * 31 + kvLen * 7);
        var q = RandomArray(seqLen * headDim, rng, 3f);
        var k = RandomArray(kvLen * headDim, rng, 3f);
        var v = RandomArray(kvLen * headDim, rng, 1f);
        var expected = new float[seqLen * headDim];
        var actual = new float[seqLen * headDim];
        Array.Fill(actual, float.NaN);
        float scale = 1f / MathF.Sqrt(headDim);

        fixed (float* pq = q, pk = k, pv = v, pe = expected, pa = actual)
        {
            AttentionKernels.ScaledDotProductFlashScalar(pq, pk, pv, pe, seqLen, kvLen, headDim, scale, causal, headDim, headDim, alibi, window);
            AttentionKernels.ScaledDotProductFlashFMA(pq, pk, pv, pa, seqLen, kvLen, headDim, scale, causal, headDim, headDim, alibi, window);
        }

        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-5f + 2e-5f * Math.Abs(expected[i]),
                $"hd={headDim} S={seqLen} KV={kvLen} causal={causal} win={window} alibi={alibi} [{i / headDim},{i % headDim}]: scalar={expected[i]} fma={actual[i]}");
    }

    /// <summary>Model-shaped: head size 128 and query/key magnitudes large enough that the softmax is peaked.</summary>
    [Theory]
    [InlineData(19, 19)]
    [InlineData(64, 576)]
    [InlineData(1, 577)]
    public unsafe void MatchesScalarFlashKernelAtModelScale(int seqLen, int kvLen)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;
        const int headDim = 128;
        var rng = new Random(99 + seqLen + kvLen);
        var q = RandomArray(seqLen * headDim, rng, 12f);
        var k = RandomArray(kvLen * headDim, rng, 12f);
        var v = RandomArray(kvLen * headDim, rng, 4f);
        var expected = new float[seqLen * headDim];
        var actual = new float[seqLen * headDim];
        float scale = 1f / MathF.Sqrt(headDim);

        fixed (float* pq = q, pk = k, pv = v, pe = expected, pa = actual)
        {
            AttentionKernels.ScaledDotProductFlashScalar(pq, pk, pv, pe, seqLen, kvLen, headDim, scale, true, headDim, headDim, 0f, 0);
            AttentionKernels.ScaledDotProductFlashFMA(pq, pk, pv, pa, seqLen, kvLen, headDim, scale, true, headDim, headDim, 0f, 0);
        }

        double maxAbs = 0;
        for (int i = 0; i < expected.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(expected[i] - actual[i]));
        Assert.True(maxAbs <= 1e-4, $"S={seqLen} KV={kvLen}: max |scalar - fma| = {maxAbs:G3}");
    }

    /// <summary>
    /// Production calls step queries and outputs by the whole layer's width, not one head's. Padded
    /// strides catch a kernel that steps by headDim instead; NaN everywhere outside the head's own
    /// output slots catches writes into a neighbouring head.
    /// </summary>
    [Fact]
    public unsafe void HonoursQueryAndOutputStridesAndWritesOnlyItsSlots()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;
        const int headDim = 60, seqLen = 7, kvLen = 21, qStride = headDim * 5 + 3, oStride = headDim * 3 + 11;
        var rng = new Random(2024);
        var q = RandomArray(seqLen * qStride, rng, 3f);
        var k = RandomArray(kvLen * headDim, rng, 3f);
        var v = RandomArray(kvLen * headDim, rng, 1f);
        var expected = new float[seqLen * oStride];
        var actual = new float[seqLen * oStride];
        Array.Fill(actual, float.NaN);
        float scale = 1f / MathF.Sqrt(headDim);

        fixed (float* pq = q, pk = k, pv = v, pe = expected, pa = actual)
        {
            AttentionKernels.ScaledDotProductFlashScalar(pq, pk, pv, pe, seqLen, kvLen, headDim, scale, true, qStride, oStride, 0f, 0);
            AttentionKernels.ScaledDotProductFlashFMA(pq, pk, pv, pa, seqLen, kvLen, headDim, scale, true, qStride, oStride, 0f, 0);
        }

        for (int i = 0; i < actual.Length; i++)
        {
            if (i % oStride < headDim)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-5f + 2e-5f * Math.Abs(expected[i]),
                    $"[{i / oStride},{i % oStride}]: scalar={expected[i]} fma={actual[i]}");
            else
                Assert.True(float.IsNaN(actual[i]), $"slot {i} outside the head's output was written: {actual[i]}");
        }
    }

    /// <summary>
    /// Rows with no visible key (a causal chunk longer than the cache) and rows whose scores are NaN
    /// take the early exits; both must still write a zero row, not leave the slot untouched.
    /// </summary>
    [Fact]
    public unsafe void RowsWithoutVisibleKeysOrWithNaNScoresAreZero()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;
        const int headDim = 16, seqLen = 5, kvLen = 2;
        var rng = new Random(8);
        var q = RandomArray(seqLen * headDim, rng, 1f);
        var k = RandomArray(kvLen * headDim, rng, 1f);
        var v = RandomArray(kvLen * headDim, rng, 1f);
        for (int d = 0; d < headDim; d++) q[4 * headDim + d] = float.NaN;   // last query: NaN scores
        var expected = new float[seqLen * headDim];
        var actual = new float[seqLen * headDim];
        Array.Fill(actual, float.NaN);

        fixed (float* pq = q, pk = k, pv = v, pe = expected, pa = actual)
        {
            AttentionKernels.ScaledDotProductFlashScalar(pq, pk, pv, pe, seqLen, kvLen, headDim, 0.25f, true, headDim, headDim, 0f, 0);
            AttentionKernels.ScaledDotProductFlashFMA(pq, pk, pv, pa, seqLen, kvLen, headDim, 0.25f, true, headDim, headDim, 0f, 0);
        }

        // queryBase = kvLen - seqLen = -3: queries 0-2 see no key; 3 sees both; 4 has NaN scores.
        for (int i = 0; i < 3 * headDim; i++) Assert.Equal(0f, actual[i]);
        for (int i = 3 * headDim; i < 4 * headDim; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-5f + 2e-5f * Math.Abs(expected[i]), $"[3,{i % headDim}]: scalar={expected[i]} fma={actual[i]}");
        for (int i = 4 * headDim; i < 5 * headDim; i++) Assert.Equal(0f, actual[i]);
    }

    private static float[] RandomArray(int n, Random rng, float range)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * range;
        return x;
    }
}
