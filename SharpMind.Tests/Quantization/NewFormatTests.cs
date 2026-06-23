using System;
using System.IO;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

public class NewFormatTests
{
    private readonly ITestOutputHelper _output;

    public NewFormatTests(ITestOutputHelper output)
    {
        _output = output;
    }

    // ── Q4_0 ──
    [Fact]
    public void TestReadQ4_0_ValidBlock()
    {
        var block = new byte[18]; // d[2] + qs[16]
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[2] = 0x34; // qs[0] = 0x34 → nibbles 4, 3
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[32];
        GgufLoader.ReadQ4_0(reader, data.AsSpan(), 32);
        Assert.Equal(-4f, data[0], 5); // (4-8)*1.0 = -4
    }

    // ── Q4_1 ──
    [Fact]
    public void TestReadQ4_1_ValidBlock()
    {
        var block = new byte[20]; // d[2] + m[2] + qs[16]
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[2] = 0x00; block[3] = 0x00; // m = 0.0
        block[4] = 0x34; // qs[0] = 0x34 → nibbles 4, 3
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[32];
        GgufLoader.ReadQ4_1(reader, data.AsSpan(), 32);
        Assert.Equal(4f, data[0], 5); // 4*1.0+0 = 4
    }

    // ── Q5_0 ──
    [Fact]
    public void TestReadQ5_0_ValidBlock()
    {
        var block = new byte[22]; // d[2] + qh[4] + qs[16]
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[6] = 0x01; // qs[0] = 0x01 → nibble 1
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[32];
        GgufLoader.ReadQ5_0(reader, data.AsSpan(), 32);
        Assert.Equal(-15f, data[0], 5); // (1-16)*1.0 = -15
    }

    // ── Q5_1 ──
    [Fact]
    public void TestReadQ5_1_ValidBlock()
    {
        var block = new byte[24]; // d[2] + m[2] + qh[4] + qs[16]
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[2] = 0x00; block[3] = 0x00; // m = 0.0
        block[8] = 0x01; // qs[0] = 0x01 → nibble 1
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[32];
        GgufLoader.ReadQ5_1(reader, data.AsSpan(), 32);
        Assert.Equal(-15f, data[0], 5); // (1-16)*1.0+0 = -15
    }

    // ── Q8_1 ──
    [Fact]
    public void TestReadQ8_1_ValidBlock()
    {
        var block = new byte[36]; // d[2] + s[2] + qs[32]
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[4] = 0x05; // qs[0] = 5
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[32];
        GgufLoader.ReadQ8_1(reader, data.AsSpan(), 32);
        Assert.Equal(5f, data[0], 5); // 5*1.0 = 5
    }

    // ── Q2_K ──
    [Fact]
    public void TestReadQ2K_ValidBlock()
    {
        var block = new byte[84]; // d[2] + dmin[2] + scales[16] + qs[64]
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[2] = 0x00; block[3] = 0x00; // dmin = 0.0
        // scales all 0, qs all 0 → val = 0
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoader.ReadQ2K(reader, data.AsSpan(), 256);
        Assert.Equal(0f, data[0], 5);
    }

    // ── Q8_K ──
    [Fact]
    public void TestReadQ8K_ValidBlock()
    {
        var block = new byte[292]; // d[4] + qs[256] + bsums[32]
        BitConverter.GetBytes(1.0f).CopyTo(block, 0); // d = 1.0f (F32)
        block[4] = 0x05; // qs[0] = 5
        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoader.ReadQ8K(reader, data.AsSpan(), 256);
        Assert.Equal(5f, data[0], 5); // 5*1.0 = 5
    }
}
