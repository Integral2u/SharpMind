using System;
using System.IO;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

public class Q3KTests
{
    private readonly ITestOutputHelper _output;

    public Q3KTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestReadQ3K_ValidBlock()
    {
        // 110-byte block: hmask(32) + qs(64) + scales(12) + d(2)
        var block = new byte[110];
        
        // d = 1.0 (0x3C00) - little endian
        block[108] = 0x00;
        block[109] = 0x3C;
        
        // scales[0..11] in bit-packed format that decodes to sc[0..15] = 0
        // buf16[0..3] low nibble, buf16[4..7] low nibble, buf16[8..11] = 0xAA gives all p[j] = 0x20, sc = 0x20-32 = 0
        for (int j = 0; j < 4; j++) block[96 + j] = 0x00;   // buf16[0..3]
        for (int j = 0; j < 4; j++) block[100 + j] = 0x00;  // buf16[4..7]
        for (int j = 0; j < 4; j++) block[104 + j] = 0xAA;  // buf16[8..11]

        // qs[0] = 0x00 (all 0s)
        block[32] = 0x00;
        
        // hmask[0] = 0x00 (all 0s)
        block[0] = 0x00;

        // Actual value = 1.0 * 0 * -4 = 0.0
        
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        
        GgufLoader.ReadQ3_K(reader, data.AsSpan(), 256);
        
        _output.WriteLine($"data[0]={data[0]}");
        
        Assert.Equal(0.0f, data[0]);
    }
}
