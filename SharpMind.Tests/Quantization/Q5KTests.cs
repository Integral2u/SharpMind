using System;
using System.IO;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

public class Q5KTests
{
    private readonly ITestOutputHelper _output;

    public Q5KTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestReadQ5_K_ValidBlock()
    {
        // 176-byte block: d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
        var block = new byte[176];
        
        // d=1.0 (0x3C00), min=0.0 (0x0000)
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;
        
        // scales[0] = 0x11 (sc=1, m=1)
        block[4] = 0x11;
        
        // qh[0] = 0x00 (hBit=0 for first 8 values)
        block[16] = 0x00;
        
        // qs[0] = 0x11 (val = 1)
        block[48] = 0x11;
        
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        
        GgufLoader.ReadQ5_K(reader, data.AsSpan(), 256);
        
        _output.WriteLine($"data[0]={data[0]}");
        
        // Actual formula: d1 * val - m1v
        // d1 = dSuper * sc0 = 1.0 * 17 = 17.0
        // val = (qs & 0xF) + (qh & u1 ? 16 : 0) = 1.0 + 0 = 1.0
        // m1v = minSuper * m = 0.0 * 0 = 0.0
        // val = 17.0 * 1 - 0.0 = 17.0
        
        Assert.Equal(17.0f, data[0]);
}

}
