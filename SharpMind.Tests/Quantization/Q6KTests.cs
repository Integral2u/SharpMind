using System;
using System.IO;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

public class Q6KTests
{
    private readonly ITestOutputHelper _output;

    public Q6KTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void TestReadQ6K_ValidBlock()
    {
        // 210-byte block: ql[128] + qh[64] + scales[16] + d[2]
        var block = new byte[210];

        // d = 1.0 (0x3C00)
        block[208] = 0x00;
        block[209] = 0x3C;

        // scales[0..15] = 0 (default)
        // ql[0..127] = 0 (default)
        // qh[0..63] = 0 (default)
        // val = d * scales[sub] * (q_val - 32) = 1.0 * 0 * (0 - 32) = 0

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];

        GgufLoader.ReadQ6K(reader, data.AsSpan(), 256);

        _output.WriteLine($"data[0]={data[0]}");

        Assert.Equal(0.0f, data[0]);
    }

    [Fact]
    public void TestReadQ6K_NonZero()
    {
        // 210-byte block: ql[128] + qh[64] + scales[16] + d[2]
        var block = new byte[210];

        // d = 1.0 (0x3C00)
        block[208] = 0x00;
        block[209] = 0x3C;

        // scales[0] = 1 (sbyte)
        block[192] = 0x01;

        // ql[0] = 0x01 → ql_val(i=0) = 1
        block[0] = 0x01;

        // qh[0] = 0x02 → qh_val(i=0) = 2
        // q_val = (2 << 4) | 1 = 33 → q_val - 32 = 1
        block[128] = 0x02;

        // val = d * scales[sub] * (q_val - 32) = 1.0 * 1 * 1 = 1.0

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];

        GgufLoader.ReadQ6K(reader, data.AsSpan(), 256);

        _output.WriteLine($"data[0]={data[0]}");

        Assert.Equal(1.0f, data[0], 5);
    }
}
