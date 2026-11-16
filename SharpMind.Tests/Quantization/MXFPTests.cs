using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpMind.Core;
using SharpMind.Core.Quantization;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// MXFP4 is a 32-value block of one E8M0 exponent and sixteen E2M1 nibble
/// pairs, and nothing downstream of the reader can sanity-check it: a shifted
/// nibble half or a scale read as fp16 still produces plausible-looking floats.
/// These fixtures are therefore hand-computed from the bit layout rather than
/// from another implementation of it.
/// </summary>
public class MXFPTests
{
    private const byte One = 127; // E8M0 127 -> 2^0

    private static QuantizationOps Ops() => QuantizationFactory.Create(HardwareTier.Scalar);

    /// <summary>
    /// One block whose sixteen bytes step the low nibble up 0..15 and the high
    /// nibble down 15..0, so a shifted half or a transposed codebook cannot pass.
    /// </summary>
    private static byte[] CheckerBlock(byte exponent)
    {
        var block = new byte[17];
        block[0] = exponent;
        for (int j = 0; j < 16; j++)
            block[1 + j] = (byte)(j | ((15 - j) << 4));
        return block;
    }

    [Fact]
    public void Read_HandComputedBlock()
    {
        var raw = CheckerBlock(One);
        var data = new float[32];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));

        Ops().ReadFor(QuantDType.MXFP4, reader, data, data.Length);

        // Low nibbles run the codebook forwards for elements 0..15 and the high
        // nibbles run it backwards for 16..31.
        float[] expected =
        [
            0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f, -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f,
            -6f, -4f, -3f, -2f, -1.5f, -1f, -0.5f, -0f, 6f, 4f, 3f, 2f, 1.5f, 1f, 0.5f, 0f,
        ];

        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], data[i]);
    }

    [Theory]
    [InlineData(127, 1f)]
    [InlineData(128, 2f)]
    [InlineData(126, 0.5f)]
    [InlineData(129, 4f)]
    [InlineData(1, 1.1754944E-38f)] // 2^-126, the smallest normal float
    [InlineData(0, 5.877472E-39f)] // 2^-127, subnormal
    public void Read_AppliesE8M0Scale(byte exponent, float scale)
    {
        var raw = CheckerBlock(exponent);
        raw[1] = 0x22; // both halves index 2 -> 1.0, so the output is exactly the scale
        var data = new float[32];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));

        Ops().ReadFor(QuantDType.MXFP4, reader, data, data.Length);

        Assert.Equal(scale, data[0]);
        Assert.Equal(scale, data[16]);
    }

    [Fact]
    public void Read_E8M0MaxIsNaN()
    {
        // 255 is the reserved E8M0 pattern. It must not decode as 2^128 (which
        // does not exist in float, so the bit shift would produce +Inf).
        var raw = CheckerBlock(255);
        var data = new float[32];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));

        Ops().ReadFor(QuantDType.MXFP4, reader, data, data.Length);

        Assert.All(data, value => Assert.True(float.IsNaN(value)));
    }

    [Fact]
    public void Read_ConsumesOneBlockPer32Values_AndTruncatesTheTail()
    {
        // 36 values is one full block plus four of the next: the second block is
        // still 17 bytes on disk, and only its first four elements may be written.
        var first = CheckerBlock(One);
        var second = new byte[17];
        second[0] = 128; // d = 2
        second[1] = 0x77; // elements 0 and 16 both index 7 -> 6.0
        second[2] = 0x77;

        var raw = new byte[34];
        first.CopyTo(raw, 0);
        second.CopyTo(raw, 17);

        var data = new float[36];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));

        Ops().ReadFor(QuantDType.MXFP4, reader, data, data.Length);

        Assert.Equal(12f, data[32]);
        Assert.Equal(12f, data[33]);
        Assert.Equal(0f, data[34]);
        Assert.Equal(0f, data[35]);
    }

    [Theory]
    [InlineData(32, 1, 17L)]
    [InlineData(64, 1, 34L)]
    [InlineData(40, 3, 102L)] // two row-padded blocks per row
    public void ByteCount_IsSeventeenPerBlockOfThirtyTwo(int row, int rows, long expected)
    {
        Assert.Equal(expected, QuantizationOps.GetRawTensorByteCount([row, rows], QuantDType.MXFP4));
        Assert.Equal(32, QuantizationOps.GetBlockSize(QuantDType.MXFP4));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(15)]
    public void E2M1BitFormula_MatchesTheCodebook(byte nibble)
    {
        // The NEON kernel decodes a nibble by building float bits instead of
        // looking one up: e > 0 is the normal float (e + 126) << 23 | m << 22,
        // e == 0 is {0, 0.5}, and bit 3 is the sign. ARM CI is the only place the
        // kernel itself runs, so the formula is checked here in scalar first.
        uint e = (uint)((nibble >> 1) & 3);
        uint m = (uint)(nibble & 1);
        uint bits = e == 0
            ? m * 0x3F000000u
            : ((e + 126) << 23) | (m << 22);
        bits |= (uint)(nibble & 8) << 28;

        float[] codebook = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f,
                            -0f, -0.5f, -1f, -1.5f, -2f, -3f, -4f, -6f];
        Assert.Equal(codebook[nibble], BitConverter.UInt32BitsToSingle(bits));
    }

    private static List<(string Name, HardwareTier Tier)> Tiers()
    {
        var tiers = new List<(string, HardwareTier)> { ("Scalar", HardwareTier.Scalar) };
        if (System.Runtime.Intrinsics.Arm.AdvSimd.Arm64.IsSupported) tiers.Add(("Arm64", HardwareTier.Arm64));
        if (System.Runtime.Intrinsics.X86.Sse.IsSupported) tiers.Add(("SSE", HardwareTier.SSE));
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported) tiers.Add(("AVX2", HardwareTier.AVX2));
        if (System.Runtime.Intrinsics.X86.Fma.IsSupported) tiers.Add(("FMA", HardwareTier.FMA));
        return tiers;
    }

    /// <summary>
    /// The dot product has to consume the same half-split the reader does. A
    /// kernel that swapped the halves, or read the scale as fp16, still returns
    /// a plausible number, so the expected value is fixed by hand: the first
    /// sixteen bytes hold low nibbles 0..7 twice and high nibbles 8..15 twice,
    /// i.e. the positive codebook in the first half of the block and the
    /// negative one in the second. A unit input on either half therefore sums to
    /// exactly ±36.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllTiers))]
    public unsafe void VecDot_HandComputedBlock(string _, HardwareTier tier)
    {
        var raw = new byte[17];
        raw[0] = One;
        for (int j = 0; j < 16; j++)
        {
            int pos = j & 7;
            raw[1 + j] = (byte)(pos | ((pos + 8) << 4));
        }

        var low = new float[32];
        var high = new float[32];
        for (int i = 0; i < 16; i++) low[i] = 1f;
        for (int i = 16; i < 32; i++) high[i] = 1f;

        var ops = QuantizationFactory.Create(tier);
        fixed (byte* pW = raw)
        {
            fixed (float* pIn = low)
                Assert.Equal(36f, ops.VecDotMXFP4(pIn, pW, 0, 32), 5);
            fixed (float* pIn = high)
                Assert.Equal(-36f, ops.VecDotMXFP4(pIn, pW, 0, 32), 5);
        }

        // ...and the scale multiplies it: E8M0 128 is 2^1.
        raw[0] = 128;
        fixed (byte* pW = raw) fixed (float* pIn = low)
            Assert.Equal(72f, ops.VecDotMXFP4(pIn, pW, 0, 32), 5);
    }

    public static IEnumerable<object[]> AllTiers()
        => Tiers().Select(t => new object[] { t.Name, t.Tier });

    /// <summary>
    /// A row whose last block is shorter than 32 values: the reader decodes the
    /// whole block, the kernels must stop at <c>inFeatures</c> and must not
    /// stride the next row's blocks off by the tail.
    /// </summary>
    [Theory]
    [InlineData(20)]
    [InlineData(33)]
    [InlineData(37)]
    [InlineData(65)]
    public unsafe void VecDot_PartialTail_MatchesReader(int inFeatures)
    {
        int nBlocks = (inFeatures + 31) / 32;
        var raw = new byte[nBlocks * 17];
        new Random(inFeatures).NextBytes(raw);
        for (int b = 0; b < nBlocks; b++) raw[b * 17] = One;

        var weights = new float[nBlocks * 32];
        using (var reader = new BinaryReader(new MemoryStream(raw, writable: false)))
            Ops().ReadFor(QuantDType.MXFP4, reader, weights, weights.Length);

        var input = new float[inFeatures];
        var rng = new Random(11);
        for (int i = 0; i < inFeatures; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        double expected = 0;
        for (int i = 0; i < inFeatures; i++) expected += input[i] * weights[i];

        foreach (var (name, tier) in Tiers())
        {
            var ops = QuantizationFactory.Create(tier);
            float actual;
            fixed (float* pIn = input) fixed (byte* pW = raw)
                actual = ops.VecDotMXFP4(pIn, pW, 0, inFeatures);
            Assert.True(Math.Abs(actual - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                $"{name}: expected {expected}, got {actual} for {inFeatures} features");
        }
    }
}
