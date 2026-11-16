using System;
using System.IO;
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
}
