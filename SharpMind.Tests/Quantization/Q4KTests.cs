using System;
using System.IO;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

public class Q4KTests
{
    private readonly ITestOutputHelper _output;

    public Q4KTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestReadQ4K_ValidBlock()
    {
        // 144-byte block: d[2] + dmin[2] + scales[12] + qs[128]
        var block = new byte[144];
        
        // d=1.0 (0x3C00), min=0.0 (0x0000)
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;
        
        // scales[0] = 0x11 (sc=1, m=1)
        block[4] = 0x11;
        
        // qs[0]=0x11 (4 values: 1, 1)
        block[16] = 0x11;
        
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        
        GgufLoader.ReadQ4K(reader, data.AsSpan(), 256);
        
        // Val = d * sc * actual = 1.0 * 17 * 1 - 0 = 17.0
        
        _output.WriteLine($"data[0]={data[0]}");
        
        Assert.Equal(17.0f, data[0]);
    }
}
