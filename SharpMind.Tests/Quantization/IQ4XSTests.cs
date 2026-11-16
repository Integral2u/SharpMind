using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// IQ4_XS is a 256-value super-block of eight 32-value sub-blocks whose 6-bit
/// scales are split across two fields, so a plausible-but-wrong scale is the
/// obvious failure and it produces plausible-looking weights. These fixtures are
/// hand-computed from the block layout (and ggml's
/// <c>dequantize_row_iq4_xs</c>), not from another decoder.
/// </summary>
public class IQ4XSTests
{
    private const int BlockBytes = 136;
    private const int BlockValues = 256;

    private static QuantizationOps Ops() => QuantizationFactory.Create(HardwareTier.Scalar);

    /// <summary>
    /// A block whose sub-block scales are 1, 2, ... 8 and whose 128 nibble bytes
    /// are all 0x88, i.e. every value indexes codebook[8] = 1.0. The decoded
    /// output is then exactly <c>d * (ls - 32)</c> per sub-block.
    /// </summary>
    private static byte[] UniformBlock()
    {
        var block = new byte[BlockBytes];
        block[0] = 0x00; block[1] = 0x3C;   // d = 1.0
        block[2] = 0x00; block[3] = 0x00;   // scales_h = 0: ls stays below 16
        block[4] = 0x21; block[5] = 0x43;   // low nibbles 1,3,5,7 -> ib 0,2,4,6
        block[6] = 0x65; block[7] = 0x87;   // high nibbles 2,4,6,8 -> ib 1,3,5,7
        block.AsSpan(8).Fill(0x88);         // index 8 in both nibble halves
        return block;
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

    public static IEnumerable<object[]> AllTiers() => Tiers().Select(t => new object[] { t.Name, t.Tier });

    [Fact]
    public void Read_DecodesTheSplitSixBitScalePerSubBlock()
    {
        var raw = UniformBlock();
        var data = new float[BlockValues];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));

        Ops().ReadFor(QuantDType.IQ4_XS, reader, data, data.Length);

        // ls 1..8 with d = 1, biased by 32, and the byte holds element j in its
        // low half (0..15) and element j + 16 in its high half.
        for (int ib = 0; ib < 8; ib++)
        {
            float expected = (ib + 1) - 32;
            for (int j = 0; j < 32; j++)
                Assert.Equal(expected, data[ib * 32 + j]);
        }
    }

    [Fact]
    public void Read_NibbleHalves_IndexTheCodebook()
    {
        // 0x9F: element 0 is codebook[15] = 113, element 16 is codebook[9] = 13.
        var raw = UniformBlock();
        raw[8] = 0x9F;
        // ib 0 keeps ls = 1, so dl = -31.
        var data = new float[BlockValues];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));

        Ops().ReadFor(QuantDType.IQ4_XS, reader, data, data.Length);

        Assert.Equal(-31f * 113f, data[0]);
        Assert.Equal(-31f * 13f, data[16]);
    }

    [Fact]
    public void Read_HighScaleBitsAreTheTopTwoOfSix()
    {
        // scales_h contributes bits 4..5: 0b11 for ib 0 gives ls = 1 | 48 = 49.
        var raw = UniformBlock();
        raw[2] = 0x03;
        var data = new float[BlockValues];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));

        Ops().ReadFor(QuantDType.IQ4_XS, reader, data, data.Length);

        Assert.Equal(49 - 32, data[0]);
        Assert.Equal(2 - 32, data[32]); // ib 1 is untouched: bits 2..3 are still 0
    }

    [Fact]
    public void Read_ConsumesOneBlockPer256Values_AndTruncatesTheTail()
    {
        var first = UniformBlock();
        var second = UniformBlock();
        second[0] = 0x00; second[1] = 0x40; // d = 2.0

        var raw = new byte[2 * BlockBytes];
        first.CopyTo(raw, 0);
        second.CopyTo(raw, BlockBytes);

        var data = new float[BlockValues + 12];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));

        Ops().ReadFor(QuantDType.IQ4_XS, reader, data, data.Length);

        // The second block starts at 256 and its ib 0 has ls = 1 -> (1-32)*2.
        Assert.Equal(-62f, data[256]);
        Assert.Equal(-62f, data[267]);
    }

    [Theory]
    [InlineData(256, 1, 136L)]
    [InlineData(512, 1, 272L)]
    [InlineData(300, 2, 544L)] // two row-padded blocks per row
    public void ByteCount_IsOneHundredThirtySixPerBlockOf256(int row, int rows, long expected)
    {
        Assert.Equal(expected, QuantizationOps.GetRawTensorByteCount([row, rows], QuantDType.IQ4_XS));
        Assert.Equal(256, QuantizationOps.GetBlockSize(QuantDType.IQ4_XS));
    }

    [Theory]
    [MemberData(nameof(AllTiers))]
    public unsafe void VecDot_HandComputedBlock(string _, HardwareTier tier)
    {
        // Every value is 1.0 (see UniformBlock) and the input is 1.0, so the dot
        // is 32 values per sub-block times the sum of the eight scales.
        var raw = UniformBlock();
        var input = new float[BlockValues];
        Array.Fill(input, 1f);

        double expected = 0;
        for (int ib = 0; ib < 8; ib++) expected += 32 * ((ib + 1) - 32);

        var ops = QuantizationFactory.Create(tier);
        fixed (float* pIn = input) fixed (byte* pW = raw)
            Assert.Equal((float)expected, ops.VecDotIQ4_XS(pIn, pW, 0, BlockValues), 4);
    }

    [Theory]
    [InlineData(300)]
    [InlineData(256 + 1)]
    [InlineData(1)]
    [InlineData(97)]
    public unsafe void VecDot_PartialTail_MatchesReader(int inFeatures)
    {
        int nBlocks = (inFeatures + BlockValues - 1) / BlockValues;
        var raw = new byte[nBlocks * BlockBytes];
        new Random(inFeatures).NextBytes(raw);
        for (int b = 0; b < nBlocks; b++)
        {
            raw[b * BlockBytes] = 0x00;
            raw[b * BlockBytes + 1] = 0x3C;
        }

        var weights = new float[nBlocks * BlockValues];
        using (var reader = new BinaryReader(new MemoryStream(raw, writable: false)))
            Ops().ReadFor(QuantDType.IQ4_XS, reader, weights, weights.Length);

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
                actual = ops.VecDotIQ4_XS(pIn, pW, 0, inFeatures);
            Assert.True(Math.Abs(actual - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                $"{name}: expected {expected}, got {actual} for {inFeatures} features");
        }
    }

    /// <summary>
    /// End-to-end: compound selection, the 136-byte shape guard and the forward
    /// kernel, for an aligned row and for one where the row is padded up to the
    /// 256-value block (the path <see cref="InferenceLinearLayer.RowPaddedKQuant"/>
    /// exists for).
    /// </summary>
    [Theory]
    [InlineData(256)]
    [InlineData(96)]
    public void Forward_MatchesDequantizedReference(int k)
    {
        const int n = 24;
        int nBlocks = (k + BlockValues - 1) / BlockValues;
        int rowBytes = nBlocks * BlockBytes;

        var raw = new byte[rowBytes * n];
        new Random(909).NextBytes(raw);
        for (int row = 0; row < n; row++)
            for (int b = 0; b < nBlocks; b++)
            {
                raw[row * rowBytes + b * BlockBytes] = 0x00;
                raw[row * rowBytes + b * BlockBytes + 1] = 0x3C;
            }

        var weights = new float[nBlocks * BlockValues * n];
        using (var reader = new BinaryReader(new MemoryStream(raw, writable: false)))
            Ops().ReadFor(QuantDType.IQ4_XS, reader, weights, weights.Length);

        var input = new float[k];
        var rng = new Random(5);
        for (int i = 0; i < k; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: false);
        using var layer = LinearLayerFactory.Create(
            "iq4_xs_proj", k, n, bias: false, weight: null, biasTensor: null, QuantDType.IQ4_XS, mapping);
        layer.SetRawWeight(raw);

        using var x = new Tensor<float>(1, k);
        input.CopyTo(x.Data);
        using var y = layer.Forward(x);

        for (int col = 0; col < n; col++)
        {
            double expected = 0;
            for (int i = 0; i < k; i++)
                expected += input[i] * weights[(long)col * nBlocks * BlockValues + i];

            float actual = y.Data[col];
            Assert.True(Math.Abs(actual - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                $"col {col}: expected {expected}, got {actual}");
        }
    }
}
