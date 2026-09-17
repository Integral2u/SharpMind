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
/// The wide Q8_0 matmul repacks eight rows per group and dots int8 activations against them.
/// Checked against a double-precision scalar reference of the same quantized arithmetic, straight
/// off the raw 34-byte blocks (catches layout, lane and scale bugs to within float accumulation),
/// and loosely against the float kernel (catches a wrong quantization scheme). Rows past the last
/// group of eight run the float kernel and are held to its own tight bound. Shapes cross the
/// seams: a tail past the last group, an odd group left without a pair, inputs above and below the
/// parallel threshold, one and many blocks. Outputs are NaN-poisoned so an unwritten slot fails.
/// </summary>
public class Q8_0WideWeightsTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (int k in new[] { 32, 96, 1024 })
            foreach (int n in new[] { 8, 13, 24, 67 })
                foreach (int m in new[] { 1, 3, 16, 33 })
                    yield return new object[] { k, n, m };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public unsafe void MatchesQuantizedReferenceAndFloatKernel(int k, int n, int m)
    {
        if (!Q8_0WideWeights.IsSupported(k, n)) return;

        var rng = new Random(4242 + k * 7 + n * 13 + m);
        var raw = RandomQ8_0(k, n, rng);
        var input = RandomInput(m * k, rng);

        var wide = Q8_0WideWeights.TryCreate(raw, k, n)!;
        var actual = new float[m * n];
        var floatRef = new float[m * n];
        Array.Fill(actual, float.NaN);
        fixed (float* pIn = input)
        fixed (float* pOut = actual)
        fixed (float* pRef = floatRef)
        fixed (byte* pW = raw)
        {
            wide.MatMul(pIn, pOut, m);
            QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar(pIn, pW, pRef, m, k, n);
        }

        int blocks = k / 32;
        int tailStart = n / 8 * 8;
        for (int p = 0; p < m; p++)
        {
            var (q, d) = QuantizeReference(input, p * k, blocks);
            for (int row = 0; row < n; row++)
            {
                double expected = 0, absWeight = 0;
                for (int b = 0; b < blocks; b++)
                {
                    int off = (row * blocks + b) * 34;
                    float dw = (float)BitConverter.UInt16BitsToHalf((ushort)(raw[off] | raw[off + 1] << 8));
                    long dot = 0;
                    for (int j = 0; j < 32; j++)
                    {
                        dot += (sbyte)raw[off + 2 + j] * q[b * 32 + j];
                        absWeight += Math.Abs((sbyte)raw[off + 2 + j] * dw);
                    }
                    expected += (double)dw * d[b] * dot;
                }

                float got = actual[p * n + row];
                float fr = floatRef[p * n + row];
                string at = $"K={k} N={n} M={m} [{p},{row}]";
                Assert.False(float.IsNaN(got), $"{at}: unwritten");
                if (row < tailStart)
                {
                    Assert.True(Math.Abs(got - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)) + 1e-2,
                        $"{at}: wide={got}, quantized reference={expected}");
                    // int8 activations round each element by at most d/2 <= max|x|/254; |x| <= 2 here.
                    Assert.True(Math.Abs(got - fr) <= absWeight * 2.0 / 254 + 1e-2,
                        $"{at}: wide={got}, float kernel={fr}");
                }
                else
                {
                    Assert.True(Math.Abs(got - fr) <= 2e-4f * Math.Max(1f, Math.Abs(fr)) + 0.05f,
                        $"{at}: tail row={got}, float kernel={fr}");
                }
            }
        }
    }

    /// <summary>
    /// A block whose maximum is too small for 127 / max to stay finite must quantize to zero. It
    /// used to overflow to ±inf, saturate to -128 and — since sign transfer cannot negate -128 —
    /// feed a wrong-signed product into the dot.
    /// </summary>
    [Fact]
    public unsafe void ZeroAndTinyActivationBlocksContributeExactlyZero()
    {
        const int k = 96, n = 8;
        if (!Q8_0WideWeights.IsSupported(k, n)) return;
        var raw = RandomQ8_0(k, n, new Random(77));
        var input = new float[k];
        for (int j = 32; j < 64; j++) input[j] = (j % 2 == 0 ? 1e-37f : -1e-37f);
        for (int j = 64; j < 96; j++) input[j] = (j % 3 == 0 ? 3e-38f : -8e-38f);

        var output = new float[n];
        var wide = Q8_0WideWeights.TryCreate(raw, k, n)!;
        fixed (float* pIn = input)
        fixed (float* pOut = output)
            wide.MatMul(pIn, pOut, 1);

        Assert.All(output, v => Assert.Equal(0f, v));
    }

    [Fact]
    public unsafe void ConcurrentCallsOnOneInstanceMatchSerialResults()
    {
        const int k = 256, n = 67, m = 16, inputs = 8;
        if (!Q8_0WideWeights.IsSupported(k, n)) return;
        var rng = new Random(31337);
        var wide = Q8_0WideWeights.TryCreate(RandomQ8_0(k, n, rng), k, n)!;
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
    public void CacheBuildsOneInstanceAndRebuildsForANewArray()
    {
        const int k = 32, n = 16;
        if (!Q8_0WideWeights.IsSupported(k, n)) return;
        var rng = new Random(5);
        var raw1 = RandomQ8_0(k, n, rng);
        var raw2 = RandomQ8_0(k, n, rng);
        var cache = new Q8_0WideWeights.Cache();

        var built = new Q8_0WideWeights?[64];
        Parallel.For(0, built.Length, i => built[i] = cache.Get(raw1, k, n));
        Assert.NotNull(built[0]);
        Assert.All(built, b => Assert.Same(built[0], b));

        var rebuilt = cache.Get(raw2, k, n);
        Assert.NotSame(built[0], rebuilt);
        Assert.Same(raw2, rebuilt!.Source);

        cache.Clear();
        Assert.False(cache.HasSlot, "Clear must drop the slot so its raw source can be collected");

        Assert.Null(new Q8_0WideWeights.Cache().Get(new byte[7 * 34], k, 7));
    }

    /// <summary>
    /// Freeing a layer must release the repack and the raw array it roots. Streaming frees and
    /// reloads a layer every forward, so a cache that outlived the free kept every layer's
    /// Q8_0 bytes resident at once — a streaming load held more than a full load. The next raw
    /// install rebuilds the repack on first use.
    /// </summary>
    [Fact]
    public void FreeFloatWeight_ReleasesTheCachedRepackAndItsRawSource()
    {
        const int k = 64, n = 16;
        if (!Q8_0WideWeights.IsSupported(k, n)) return;
        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);
        var layer = (InferenceLinearLayer)LinearLayerFactory.Create("wide", k, n, false, null, null, QuantDType.Q8_0, mapping);
        Assert.True(layer.WideAllowed);

        layer.SetRawWeight(RandomQ8_0(k, n, new Random(9)));
        using (var input = new Tensor<float>(1, k))
        using (layer.Forward(input)) { }
        Assert.True(layer.HasCachedWide, "the first forward must build the repack");

        layer.FreeFloatWeight();
        Assert.False(layer.HasCachedWide, "freeing the layer must drop the repack and its raw source");

        layer.Dispose();
    }

    /// <summary>The wide path stands in for the parallel FMA Q8_0 kernel only: a serial selection keeps its kernel.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void OnlyTheParallelFmaSelectionAllowsTheWidePath(bool parallel)
    {
        if (!Q8_0WideWeights.IsSupported(64, 64)) return;
        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: parallel);

        var q8 = (InferenceLinearLayer)LinearLayerFactory.Create("q8", 64, 64, false, null, null, QuantDType.Q8_0, mapping);
        Assert.Equal(parallel, q8.WideAllowed);

        var head = LogitOpsFactory.Create(new Tensor<float>(1, 1), new byte[64 * 2 * 34], QuantDType.Q8_0, mapping);
        Assert.Equal(parallel, head.WideAllowed);

        var f32 = (InferenceLinearLayer)LinearLayerFactory.Create("f32", 64, 64, false, null, null, QuantDType.F32, mapping);
        Assert.False(f32.WideAllowed);
    }

    [Fact]
    public void UnsupportedShapesAreRefused()
    {
        Assert.Null(Q8_0WideWeights.TryCreate(new byte[7 * 34], 32, 7));
        Assert.Null(Q8_0WideWeights.TryCreate(new byte[8 * 2 * 34], 48, 8));
        Assert.Null(Q8_0WideWeights.TryCreate(new byte[8 * 34 - 1], 32, 8));
    }

    private static byte[] RandomQ8_0(int k, int n, Random rng)
    {
        var raw = new byte[(int)QuantizationOps.GetRawTensorByteCount([n, k], QuantDType.Q8_0)];
        rng.NextBytes(raw);
        for (int off = 0; off < raw.Length; off += 34)
        {
            ushort half = BitConverter.HalfToUInt16Bits((Half)(0.25 + rng.NextDouble()));
            raw[off] = (byte)half;
            raw[off + 1] = (byte)(half >> 8);
        }
        return raw;
    }

    private static float[] RandomInput(int length, Random rng)
    {
        var x = new float[length];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        return x;
    }

    private static (int[] Q, float[] D) QuantizeReference(float[] input, int offset, int blocks)
    {
        var q = new int[blocks * 32];
        var d = new float[blocks];
        for (int b = 0; b < blocks; b++)
        {
            float amax = 0;
            for (int j = 0; j < 32; j++) amax = MathF.Max(amax, MathF.Abs(input[offset + b * 32 + j]));
            d[b] = amax / 127f;
            float id = amax >= 1e-30f ? 127f / amax : 0f;
            for (int j = 0; j < 32; j++)
                q[b * 32 + j] = (int)MathF.Round(input[offset + b * 32 + j] * id, MidpointRounding.ToEven);
        }
        return (q, d);
    }

    private static unsafe float[] Run(Q8_0WideWeights wide, float[] input, int m, int n)
    {
        var output = new float[m * n];
        fixed (float* pIn = input)
        fixed (float* pOut = output)
            wide.MatMul(pIn, pOut, m);
        return output;
    }
}
