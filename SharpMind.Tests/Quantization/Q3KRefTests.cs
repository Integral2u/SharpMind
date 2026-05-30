using System;
using System.IO;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// Reference tests for Q3_K dequant, matching GGUF block_q3_K layout:
///   hmask[32] + qs[64] + scales[12] + d[2] = 110 bytes
///   hmask @ 0, qs @ 32, scales @ 96, d @ 108
///
/// Dequant formula: result = d * (scales[sub-block] - 32) * (qs_val - (hmask_bit ? 0 : 4))
/// </summary>
public class Q3KRefTests
{
    private readonly ITestOutputHelper _output;

    public Q3KRefTests(ITestOutputHelper output) { _output = output; }

    private static byte[] MakeBlock()
    {
        var block = new byte[110];
        block[108] = 0x00; block[109] = 0x3C; // d = 1.0
        return block;
    }

    private static float[] Dequant(byte[] block, int count = 256)
    {
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[count];
        GgufLoader.ReadQ3_K(reader, data.AsSpan(), count);
        return data;
    }

    [Fact]
    public void TestQ3K_Scale0_32()
    {
        var block = MakeBlock();

        block[96] = 0x00; // scale[0] low nibble = 0
        block[104] = 0x02; // scale[0] high nibble = 2

        var data = Dequant(block);

        Assert.Equal(0.0f, data[0]);
        Assert.Equal(0.0f, data[15]);
        Assert.Equal(128.0f, data[16]);
        Assert.Equal(128.0f, data[31]);
    }

    [Fact]
    public void TestQ3K_HighBitSet()
    {
        var block = MakeBlock();
        block[96] = 0x00;
        block[104] = 0x02;

        block[32 + 16] = 0x01; // qs[16] = 0x01 → qs_val = low 2 bits = 1
        block[0 + 16] = 0x01;  // hmask[16] bit 0 = 1 → subtract 0

        var data = Dequant(block);

        Assert.Equal(0.0f, data[0]);
        Assert.Equal(-32.0f, data[16]);
    }

    [Fact]
    public void TestQ3K_HighBitClear()
    {
        var block = MakeBlock();
        block[96] = 0x00;
        block[104] = 0x02;

        block[32 + 16] = 0x01;
        block[0 + 16] = 0x00; // hmask[16] bit 0 = 0 → subtract 4

        var data = Dequant(block);

        Assert.Equal(96.0f, data[16]);
    }

    [Fact]
    public void TestQ3K_Shift2()
    {
        var block = MakeBlock();
        block[96] = 0x00;
        block[104] = 0x02;

        block[32] = 0x0C; // qs[0] bits 2-3 = 3
        block[0] = 0x02;  // hmask[0] bit 1 = 1

        var data = Dequant(block);

        Assert.Equal(-96.0f, data[32]);
    }

    [Fact]
    public void TestQ3K_Scale5_16()
    {
        var block = MakeBlock();
        block[96] = 0x00;
        block[104] = 0x02;

        block[97] = 0x00; // scale[5] low nibble = 0 at buf[97]
        block[105] = 0x04; // scale[5] high nibble = 1 at buf[105]

        var data = Dequant(block);

        Assert.Equal(0.0f, data[0]);
        Assert.Equal(64.0f, data[80]);
    }

}
