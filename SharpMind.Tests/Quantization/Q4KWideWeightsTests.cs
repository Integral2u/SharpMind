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
/// The wide Q4_K matmul keeps 4-bit weights packed, repacks eight rows per group,
/// and evaluates each 32-value sub-block as
/// <c>scale * dot(v, q) - bias * sum(q)</c> against int8 activations.
/// Checked against a double-precision reference of the same arithmetic straight
/// off the raw 144-byte blocks, and against the float kernel for the tail rows
/// that still take the old path.
/// </summary>
public sealed class Q4KWideWeightsTests
{
    private const int SuperBlock = 256;
    private const int RawBlockBytes = 144;

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
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(9001 + k * 3 + n * 7 + m);
        var raw = RandomQ4K(k, n, rng);
        var input = RandomInput(m * k, rng);

        var wide = Q4KWideWeights.TryCreate(raw, k, n)!;
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
                    Math.Abs(got - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)) + 1e-2,
                    $"{at}: wide={got}, reference={expected}");
            }
        }
    }

    [Fact]
    public unsafe void TailRowsMatchTheFloatKernel()
    {
        const int k = 512, n = 13, m = 2;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(123);
        var raw = RandomQ4K(k, n, rng);
        var input = RandomInput(m * k, rng);
        var wide = Q4KWideWeights.TryCreate(raw, k, n)!;
        var actual = new float[m * n];
        var reference = new float[m * n];
        Array.Fill(actual, float.NaN);

        fixed (float* pIn = input)
        fixed (float* pOut = actual)
        fixed (float* pRef = reference)
        fixed (byte* pRaw = raw)
        {
            wide.MatMul(pIn, pOut, m);
            QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar(pIn, pRaw, pRef, m, k, n);
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
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        var wide = Q4KWideWeights.TryCreate(RandomQ4K(k, n, new Random(77)), k, n)!;
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
        Assert.Null(Q4KWideWeights.TryCreate(new byte[8 * RawBlockBytes], 128, 8));
        Assert.Null(Q4KWideWeights.TryCreate(new byte[7 * RawBlockBytes], 256, 7));
        Assert.Null(Q4KWideWeights.TryCreate(new byte[8 * RawBlockBytes - 1], 256, 8));
        Assert.Null(Q4KWideWeights.TryCreate(new byte[8 * 2 * RawBlockBytes - 1], 512, 8));
    }

    [Fact]
    public void CacheBuildsOneInstanceAndRebuildsForANewArray()
    {
        const int k = 256, n = 16;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(5);
        var raw1 = RandomQ4K(k, n, rng);
        var raw2 = RandomQ4K(k, n, rng);
        var cache = new Q4KWideWeights.Cache();

        var built = new Q4KWideWeights?[64];
        Parallel.For(0, built.Length, i => built[i] = cache.Get(raw1, k, n));
        Assert.NotNull(built[0]);
        Assert.All(built, value => Assert.Same(built[0], value));

        var rebuilt = cache.Get(raw2, k, n);
        Assert.NotSame(built[0], rebuilt);
        Assert.Same(raw2, rebuilt!.Source);

        cache.Clear();
        Assert.False(cache.HasSlot, "Clear must drop the slot so its raw source can be collected");
    }

    [Fact]
    public unsafe void ConcurrentCallsOnOneInstanceMatchSerialResults()
    {
        const int k = 256, n = 24, m = 8, inputs = 8;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(31337);
        var wide = Q4KWideWeights.TryCreate(RandomQ4K(k, n, rng), k, n)!;
        var xs = new float[inputs][];
        var expected = new float[inputs][];
        for (int t = 0; t < inputs; t++)
        {
            xs[t] = RandomInput(m * k, rng);
            expected[t] = Run(wide, xs[t], m, n);
        }

        Parallel.For(0, inputs * 6, new ParallelOptions { MaxDegreeOfParallelism = inputs }, i =>
            Assert.Equal(expected[i % inputs], Run(wide, xs[i % inputs], m, n)));
    }

    [Fact]
    public unsafe void LayerForwardUsesTheRepackAndFreeDropsIt()
    {
        const int k = 256, n = 16;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);
        var layer = (InferenceLinearLayer)LinearLayerFactory.Create(
            "q4", k, n, false, null, null, QuantDType.Q4_K, mapping);
        Assert.True(layer.WideAllowed);

        layer.SetRawWeight(RandomQ4K(k, n, new Random(9)));
        var input = RandomInput(k, new Random(10));
        using var inputTensor = new Tensor<float>(1, k);
        input.CopyTo(inputTensor.Data);
        using (var output = layer.Forward(inputTensor))
        {
            Assert.Equal(n, output.ElementCount);
        }

        Assert.True(layer.HasCachedQ4KWide, "the first forward must build the Q4_K repack");
        layer.FreeFloatWeight();
        Assert.False(layer.HasCachedQ4KWide, "freeing the layer must drop the repack");
        layer.Dispose();
    }

    [Fact]
    public unsafe void LogitOpsProjectUsesTheRepack()
    {
        const int k = 256, n = 16;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        var rng = new Random(11);
        var raw = RandomQ4K(k, n, rng);
        var input = RandomInput(k, rng);
        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);
        var ops = LogitOpsFactory.Create(null, raw, QuantDType.Q4_K, mapping);
        Assert.True(ops.WideAllowed);

        using var inputTensor = new Tensor<float>(1, k);
        input.CopyTo(inputTensor.Data);
        using var projected = ops.Project(inputTensor, 1, k, n);

        // Project must select the same wide kernel, so its output is identical
        // to calling that kernel directly with the same input.
        var wide = Q4KWideWeights.TryCreate(raw, k, n)!;
        var reference = new float[n];
        fixed (float* pIn = input)
        fixed (float* pRef = reference)
            wide.MatMul(pIn, pRef, 1);

        Assert.Equal(reference, projected.Data.ToArray());
    }

    private static byte[] RandomQ4K(int k, int n, Random rng)
    {
        int blocks = k / SuperBlock;
        var raw = new byte[(long)n * blocks * RawBlockBytes];
        rng.NextBytes(raw);
        for (int off = 0; off < raw.Length; off += RawBlockBytes)
        {
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.25 + rng.NextDouble()));
            ushort dmin = BitConverter.HalfToUInt16Bits((Half)(0.05 + rng.NextDouble() * 0.1));
            raw[off] = (byte)d;
            raw[off + 1] = (byte)(d >> 8);
            raw[off + 2] = (byte)dmin;
            raw[off + 3] = (byte)(dmin >> 8);
        }
        return raw;
    }

    private static float[] RandomInput(int length, Random rng)
    {
        var x = new float[length];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        return x;
    }

    private static (int[] Q, float[] D) QuantizeReference(float[] input, int offset, int blockCount)
    {
        var q = new int[blockCount * 32];
        var d = new float[blockCount];
        for (int b = 0; b < blockCount; b++)
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
                float d = (float)BitConverter.UInt16BitsToHalf(UnsafeReadUInt16(block));
                float dmin = (float)BitConverter.UInt16BitsToHalf(UnsafeReadUInt16(block + 2));
                byte* scales = block + 4;
                byte* qs = block + 16;

                for (int sub = 0; sub < 8; sub++)
                {
                    int sc = QuantizationKernels.GetScaleMinK4_Scale_Scalar(sub, scales);
                    int mn = QuantizationKernels.GetScaleMinK4_Min_Scalar(sub, scales);
                    int chunk = sub / 2;
                    int shift = (sub % 2) * 4;
                    int dot = 0, sumQ = 0;
                    int qBase = (b * 8 + sub) * 32;
                    for (int i = 0; i < 32; i++)
                    {
                        int v = (qs[chunk * 32 + i] >> shift) & 0x0F;
                        dot += v * q[qBase + i];
                        sumQ += q[qBase + i];
                    }
                    sum += dq[b * 8 + sub] * (d * sc * dot - dmin * mn * sumQ);
                }
            }
        }
        return sum;
    }

    private static unsafe ushort UnsafeReadUInt16(byte* p)
        => (ushort)(p[0] | (p[1] << 8));

    private static unsafe float[] Run(Q4KWideWeights wide, float[] input, int m, int n)
    {
        var output = new float[m * n];
        fixed (float* pIn = input)
        fixed (float* pOut = output)
            wide.MatMul(pIn, pOut, m);
        return output;
    }
}
