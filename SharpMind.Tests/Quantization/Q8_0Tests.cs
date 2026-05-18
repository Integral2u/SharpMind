using System;
using System.IO;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

public class Q8_0Tests
{
    private readonly ITestOutputHelper _output;

    public Q8_0Tests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestReadQ8_0_ValidBlock()
    {
        // 34-byte block: d[2] + sbyte[32]
        var block = new byte[34];

        // d = 1.0 (0x3C00)
        block[0] = 0x00;
        block[1] = 0x3C;

        // sbyte[0] = 0 (default)
        // val = 0 * 1.0 = 0

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[32];
        
        GgufLoader.ReadQ8_0(reader, data.AsSpan(), 32);

        _output.WriteLine($"data[0]={data[0]}");

        Assert.Equal(0.0f, data[0]);
    }

    [Fact]
    public void TestReadQ8_0_NonZero()
    {
        // 34-byte block: d[2] + sbyte[32]
        var block = new byte[34];

        // d = 2.0 (0x4000)
        block[0] = 0x00;
        block[1] = 0x40;

        // sbyte[0] = 3
        block[2] = 0x03;

        // val = 3 * 2.0 = 6.0

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[32];
        
        GgufLoader.ReadQ8_0(reader, data.AsSpan(), 32);

        _output.WriteLine($"data[0]={data[0]}");

        Assert.Equal(6.0f, data[0], 5);
    }
}
