using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// The wide Q6_K matmul widens the 6-bit weights to bytes during repack, groups
/// eight rows, and evaluates each 16-value sub-block as
/// <c>scale * (Σ(q6*q) - 32*Σq)</c> against int8 activations. Checked against a
/// double-precision reference of the same arithmetic straight off the raw 210-byte
/// blocks, and against the float kernel for tail rows.
/// </summary>
public sealed class Q6KWideWeightsTests
{
    private const int SuperBlock = 256;
    private const int RawBlockBytes = 210;

    public static IEnumerable<object[]> Cases()
    {
        foreach (int k in new[] { 256, 512, 1024, 2048 })
            foreach (int n in new[] { 8, 16, 24, 40 })
                foreach (int m in new[] { 1, 3, 16 })
                    yield return new object[] { k, n, m };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public unsafe void MatchesGroupedReference(int k, int n, int m)
    {
        if (!Q6KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(7001 + k * 3 + n * 7 + m);
        var raw = RandomQ6K(k, n, rng);
        var input = RandomInput(m * k, rng);

        var wide = Q6KWideWeights.TryCreate(raw, k, n)!;
        var actual = new float[m * n];
        Array.Fill(actual, float.NaN);
        fixed (float* pIn = input)
        fixed (float* pOut = actual)
            wide.MatMul(pIn, pOut, m);

        for (int p = 0; p < m; p++)
        {
            var (q, dq) = QuantizeReference(input, p * k, k / 32);
            for (int row = 0; row < n; row++)
            {
                double expected = ReferenceRow(raw, row, k, q, dq);
                float got = actual[p * n + row];
                string at = $"K={k} N={n} M={m} [{p},{row}]";
                Assert.False(float.IsNaN(got), $"{at}: unwritten");
                Assert.True(
                    Math.Abs(got - expected) <= 2e-4 * Math.Max(1, Math.Abs(expected)) + 2e-2,
                    $"{at}: wide={got}, reference={expected}");
            }
        }
    }

    [Fact]
    public unsafe void TailRowsMatchTheFloatKernel()
    {
        const int k = 512, n = 13, m = 2;
        if (!Q6KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(123);
        var raw = RandomQ6K(k, n, rng);
        var input = RandomInput(m * k, rng);
        var wide = Q6KWideWeights.TryCreate(raw, k, n)!;
        var actual = new float[m * n];
        var reference = new float[m * n];
        Array.Fill(actual, float.NaN);

        fixed (float* pIn = input)
        fixed (float* pOut = actual)
        fixed (float* pRef = reference)
        fixed (byte* pRaw = raw)
        {
            wide.MatMul(pIn, pOut, m);
            QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar(pIn, pRaw, pRef, m, k, n);
        }

        for (int row = 8; row < n; row++)
        {
            for (int p = 0; p < m; p++)
            {
                float got = actual[p * n + row];
                float want = reference[p * n + row];
                Assert.True(
                    Math.Abs(got - want) <= 2e-4f * Math.Max(1f, Math.Abs(want)) + 0.05f,
                    $"tail [{p},{row}]: wide={got}, float kernel={want}");
            }
        }
    }

    [Fact]
    public unsafe void ZeroActivationGivesZeroOutput()
    {
        const int k = 256, n = 8;
        if (!Q6KWideWeights.IsSupported(k, n)) return;

        var wide = Q6KWideWeights.TryCreate(RandomQ6K(k, n, new Random(77)), k, n)!;
        var input = new float[k];
        var output = new float[n];
        Array.Fill(output, float.NaN);

        fixed (float* pIn = input)
        fixed (float* pOut = output)
            wide.MatMul(pIn, pOut, 1);

        Assert.All(output, value => Assert.Equal(0f, value));
    }

    [Fact]
    public void UnsupportedShapesAreRefused()
    {
        Assert.Null(Q6KWideWeights.TryCreate(new byte[8 * RawBlockBytes], 128, 8));
        Assert.Null(Q6KWideWeights.TryCreate(new byte[7 * RawBlockBytes], 256, 7));
        Assert.Null(Q6KWideWeights.TryCreate(new byte[8 * RawBlockBytes - 1], 256, 8));
        Assert.Null(Q6KWideWeights.TryCreate(new byte[8 * 2 * RawBlockBytes - 1], 512, 8));
    }

    [Fact]
    public void CacheBuildsOneInstanceAndRebuildsForANewArray()
    {
        const int k = 256, n = 16;
        if (!Q6KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(5);
        var raw1 = RandomQ6K(k, n, rng);
        var raw2 = RandomQ6K(k, n, rng);
        var cache = new Q6KWideWeights.Cache();

        var built = new Q6KWideWeights?[64];
        Parallel.For(0, built.Length, i => built[i] = cache.Get(raw1, k, n));
        Assert.NotNull(built[0]);
        Assert.All(built, value => Assert.Same(built[0], value));

        var rebuilt = cache.Get(raw2, k, n);
        Assert.NotSame(built[0], rebuilt);
        Assert.Same(raw2, rebuilt!.Source);

        cache.Clear();
        Assert.False(cache.HasSlot);
    }

    [Fact]
    public unsafe void LayerAndLogitOpsUseTheRepack()
    {
        const int k = 256, n = 16;
        if (!Q6KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(11);
        var raw = RandomQ6K(k, n, rng);
        var input = RandomInput(k, rng);
        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);

        var layer = (InferenceLinearLayer)LinearLayerFactory.Create(
            "q6", k, n, false, null, null, QuantDType.Q6_K, mapping);
        Assert.True(layer.WideAllowed);
        layer.SetRawWeight(raw);

        using var inputTensor = new Tensor<float>(1, k);
        input.CopyTo(inputTensor.Data);
        using (layer.Forward(inputTensor))
        {
        }
        Assert.True(layer.HasCachedQ6KWide);
        layer.FreeFloatWeight();
        Assert.False(layer.HasCachedQ6KWide);
        layer.Dispose();

        var ops = LogitOpsFactory.Create(null, raw, QuantDType.Q6_K, mapping);
        Assert.True(ops.WideAllowed);
        using var projected = ops.Project(inputTensor, 1, k, n);

        var wide = Q6KWideWeights.TryCreate(raw, k, n)!;
        var reference = new float[n];
        fixed (float* pIn = input)
        fixed (float* pRef = reference)
            wide.MatMul(pIn, pRef, 1);

        Assert.Equal(reference, projected.Data.ToArray());
    }

    private static byte[] RandomQ6K(int k, int n, Random rng)
    {
        int blocks = k / SuperBlock;
        var raw = new byte[(long)n * blocks * RawBlockBytes];
        rng.NextBytes(raw);
        for (int off = 0; off < raw.Length; off += RawBlockBytes)
        {
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.25 + rng.NextDouble()));
            raw[off + 208] = (byte)d;
            raw[off + 209] = (byte)(d >> 8);
        }
        return raw;
    }

    private static float[] RandomInput(int length, Random rng)
    {
        var x = new float[length];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        return x;
    }

    private static (int[] Q, float[] D) QuantizeReference(float[] input, int offset, int blocks32)
    {
        var q = new int[blocks32 * 32];
        var d = new float[blocks32];
        for (int b = 0; b < blocks32; b++)
        {
            float amax = 0;
            for (int j = 0; j < 32; j++) amax = MathF.Max(amax, MathF.Abs(input[offset + b * 32 + j]));
            d[b] = amax / 127f;
            float id = amax >= 1e-30f ? 127f / amax : 0f;
            for (int j = 0; j < 32; j++)
                q[b * 32 + j] = (int)MathF.Round(
                    input[offset + b * 32 + j] * id, MidpointRounding.ToEven);
        }
        return (q, d);
    }

    private static unsafe double ReferenceRow(byte[] raw, int row, int k, int[] q, float[] dq)
    {
        int blocks = k / SuperBlock;
        double sum = 0;
        fixed (byte* pRaw = raw)
        {
            for (int b = 0; b < blocks; b++)
            {
                byte* block = pRaw + ((long)row * blocks + b) * RawBlockBytes;
                byte* ql = block;
                byte* qh = block + 128;
                sbyte* scales = (sbyte*)(block + 192);
                float d = (float)BitConverter.UInt16BitsToHalf(UnsafeReadUInt16(block + 208));

                for (int half = 0; half < 2; half++)
                {
                    byte* pql = ql + half * 64;
                    byte* pqh = qh + half * 32;
                    sbyte* psc = scales + half * 8;

                    for (int l = 0; l < 32; l++)
                    {
                        int group = l / 16;
                        int q1 = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                        int q2 = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                        int q3 = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                        int q4 = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                        int baseIdx = b * SuperBlock + half * 128 + l;
                        sum += Term(q, dq, baseIdx, psc[group], d, q1);
                        sum += Term(q, dq, baseIdx + 32, psc[group + 2], d, q2);
                        sum += Term(q, dq, baseIdx + 64, psc[group + 4], d, q3);
                        sum += Term(q, dq, baseIdx + 96, psc[group + 6], d, q4);
                    }
                }
            }
        }
        return sum;
    }

    private static double Term(int[] q, float[] dq, int index, int scale, float d, int value)
    {
        int activation = q[index];
        float weight = (float)(Half)(d * scale);
        return dq[index / 32] * weight * (value * activation - 32 * activation);
    }

    private static unsafe ushort UnsafeReadUInt16(byte* p)
        => (ushort)(p[0] | (p[1] << 8));
}
