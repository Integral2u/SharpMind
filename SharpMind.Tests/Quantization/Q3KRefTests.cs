using System;
using System.IO;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// Reference tests for Q3_K dequant, matching llama.cpp block_q3_K layout:
///   d[2] + dmin[2] + hmask[32] + qs[64] + scales[12] = 112 bytes
///
/// Scale packing: 12 bytes at offset 100 → 16 int8 scales.
/// Reassembly (C dequantize_row_q3_K):
///   aux[0] = bytes 100-103 (low nibbles of scales[0..3], scales[8..11])
///   aux[1] = bytes 104-107 (low nibbles of scales[4..7], scales[12..15])
///   tmp    = bytes 108-111 (each byte: bits 0-1=high{0,1,2,3}, 2-3=high{4,5,6,7},
///                          4-5=high{8,9,10,11}, 6-7=high{12,13,14,15})
///   aux[0] = (aux[0] & 0x0f0f0f0f) | (((tmp >> 0) & 0x03030303) << 4)
///   aux[1] = (aux[1] & 0x0f0f0f0f) | (((tmp >> 2) & 0x03030303) << 4)
///   aux[2] = ((aux[0] >> 4) & 0x0f0f0f0f) | (((tmp >> 4) & 0x03030303) << 4)
///   aux[3] = ((aux[1] >> 4) & 0x0f0f0f0f) | (((tmp >> 6) & 0x03030303) << 4)
///
/// Dequant formula: result = d * (scales[sub-block] - 32) * (qs_val - (hmask_bit ? 0 : 4))
///   where qs_val = (qs[qs_byte] >> qs_shift) & 3
///   qs_byte = (i/128)*32 + (i%32), qs_shift = ((i%128)/32)*2
///   hmask_bit = (hmask[i%32] >> (i/32)) & 1
/// </summary>
public class Q3KRefTests
{
    private readonly ITestOutputHelper _output;

    public Q3KRefTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// scale[0]=32: low nibble 0 at buf[100] byte 0 low nibble, high nibble 2 at buf[108] bits 0-1
    ///   block[100] = 0x00 (byte 0 low nibble = 0)
    ///   block[108] = 0x02 (bits 0-1 = 2)
    /// scale[0]=32: dl = d*(32-32)=0 → elements 0..15 = 0
    /// All other scales=0: dl = 1*(0-32) = -32
    /// qs=0, hmask=0 → val = -32 * (0-4) = 128
    /// </summary>
    [Fact]
    public void TestQ3K_Scale0_32()
    {
        var block = new byte[112];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0

        block[100] = 0x00; // scale[0] low nibble = 0
        block[108] = 0x02; // scale[0] high nibble = 2

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoader.ReadQ3_K(reader, data.AsSpan(), 256);

        Assert.Equal(0.0f, data[0]);
        Assert.Equal(0.0f, data[15]);
        Assert.Equal(128.0f, data[16]);
        Assert.Equal(128.0f, data[31]);
    }

    /// <summary>
    /// scale[0]=32 → dl=0 for elements 0..15.
    /// scale[1]=0 → dl=-32 for elements 16..31.
    /// element 16: qs[0..15] bits 0-1 at (16/128)*32 + (16%32) = 0*32+16 = 16 → qs[16] bits 0-1
    ///   set qs[36+16]=0x01 → bits 0-1 = 1 → qs_val=1
    ///   hmask[4+16]=0x01 → bit 0 = 1 → subtract 0
    ///   val = -32 * (1-0) = -32
    /// </summary>
    [Fact]
    public void TestQ3K_HighBitSet()
    {
        var block = new byte[112];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[100] = 0x00; // scale[0] low nibble = 0
        block[108] = 0x02; // scale[0] high nibble = 2 → scale[0]=32, rest=0

        block[36 + 16] = 0x01; // qs[16] = 0x01 → qs_val = low 2 bits = 1
        block[4 + 16] = 0x01;  // hmask[16] bit 0 = 1 → subtract 0

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoader.ReadQ3_K(reader, data.AsSpan(), 256);

        Assert.Equal(0.0f, data[0]);
        Assert.Equal(-32.0f, data[16]);
    }

    /// <summary>
    /// Same as HighBitSet but hmask cleared → val = -32 * (1-4) = 96
    /// </summary>
    [Fact]
    public void TestQ3K_HighBitClear()
    {
        var block = new byte[112];
        block[0] = 0x00; block[1] = 0x3C;
        block[100] = 0x00;
        block[108] = 0x02;

        block[36 + 16] = 0x01;
        block[4 + 16] = 0x00; // hmask[16] bit 0 = 0 → subtract 4

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoader.ReadQ3_K(reader, data.AsSpan(), 256);

        Assert.Equal(96.0f, data[16]);
    }

    /// <summary>
    /// Element 32: i=32 → qs_byte = (32/128)*32 + (32%32) = 0+0 = 0, qs_shift = ((32%128)/32)*2 = (32/32)*2 = 2
    ///   qs[0] bits 2-3 → set qs[36+0]=0x0C → bits 2-3 = 3
    ///   hmask_bit = (hmask[32%32] >> (32/32)) & 1 = (hmask[0] >> 1) & 1
    ///   set block[4+0]=0x02 → bit 1 = 1
    ///   val = -32 * (3-0) = -96
    /// </summary>
    [Fact]
    public void TestQ3K_Shift2()
    {
        var block = new byte[112];
        block[0] = 0x00; block[1] = 0x3C;
        block[100] = 0x00;
        block[108] = 0x02;

        block[36] = 0x0C; // qs[0] bits 2-3 = 3
        block[4] = 0x02;  // hmask[0] bit 1 = 1

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoader.ReadQ3_K(reader, data.AsSpan(), 256);

        Assert.Equal(-96.0f, data[32]);
    }

    /// <summary>
    /// scale[5]=16: i=5 → aux index = (5-4)=1 → byte 1 of aux[1]
    ///   low nibble = buf[101] & 0x0f = 0
    ///   high nibble = (buf[109] >> 2) & 0x03 = 1
    ///   value = (1 << 4) | 0 = 16
    /// scale[5]=16: dl = 1*(16-32) = -16
    /// Element 80 = 5*16 = 80. qs_val=0, hmask_bit=0 → -16 * (0-4) = 64
    /// </summary>
    [Fact]
    public void TestQ3K_Scale5_16()
    {
        var block = new byte[112];
        block[0] = 0x00; block[1] = 0x3C;
        block[100] = 0x00;
        block[108] = 0x02;

        block[101] = 0x00; // scale[5] low nibble = 0 at buf[101] low 4 bits
        block[109] = 0x04; // scale[5] high nibble = 1 at buf[109] bits 2-3

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoader.ReadQ3_K(reader, data.AsSpan(), 256);

        Assert.Equal(0.0f, data[0]);
        Assert.Equal(64.0f, data[80]);
    }
}
