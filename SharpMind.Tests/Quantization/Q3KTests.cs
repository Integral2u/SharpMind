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
        // 110-byte block: hmask[32] + qs[64] + scales[12] + d[2]
        var block = new byte[110];
        
        // d = 1.0 (0x3C00) at bytes 108-109
        block[108] = 0x00;
        block[109] = 0x3C;
        
        // All hmask = 0, all qs = 0, all scales = 0
        // val = d * (0 - 32) * (0 - 4) = 1.0 * (-32) * (-4) = 128.0
        
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        
        GgufLoaderFactory.Default.ReadQ3_K(reader, data.AsSpan(), 256);
        
        _output.WriteLine($"data[0]={data[0]}");
        
        Assert.Equal(128.0f, data[0]);
    }
}
