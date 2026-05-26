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
        
        // d = 17.0 (fp16 0x4C40) at bytes 0-1, dmin = 0.0 at bytes 2-3
        block[0] = 0x40; block[1] = 0x4C;
        block[2] = 0x00; block[3] = 0x00;
        
        // scales[0] = 0x11 → sc=17, scales[4] = 0x00 → m=0
        block[4] = 0x11;
        
        // qh[0] bit 0 = 1 (u1=1 for first iteration)
        block[16] = 0x01;
        
        // qs[0] = 0x01 → low nibble = 1
        block[48] = 0x01;
        
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        
        GgufLoader.ReadQ5_K(reader, data.AsSpan(), 256);
        
        _output.WriteLine($"data[0]={data[0]}");
        
        // Actual formula: d1 * qv - m1v
        // d1 = d * sc0 = 17.0 * 17 = 289.0
        // qv = (qs & 0x0F) + (qh & u1 ? 16 : 0) = 1 + 16 = 17
        // data[0] = 289.0 * 17 - 0 = 4913.0
        
        Assert.Equal(4913.0f, data[0]);
}

}
