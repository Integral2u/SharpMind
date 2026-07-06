using System;
using System.IO;
using SharpMind.Model.Format;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

public class Q6KRefTests
{
    private readonly ITestOutputHelper _output;

    public Q6KRefTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Q6_K block: ql[128] + qh[64] + sc[16] + d[2] = 210 bytes
    /// Each element is a 6-bit signed value centered at 32.
    /// value = d * sc[idx] * (q6 - 32)
    ///
    /// Set d=1.0, all ql/qh=0, sc[0]=32, rest sc=0
    /// Elements 0..15: sc[0]=32 → d*32*(0-32) = -1024
    /// Elements 16..31: sc[1]=0 → d*0*(0-32) = 0
    /// </summary>
    [Fact]
    public void TestQ6K_Basic()
    {
        var block = new byte[210];
        // d = 1.0 at offset 208
        block[208] = 0x00; block[209] = 0x3C;

        // sc at offset 192: 16 signed bytes
        // sc[0] = 32, rest = 0
        block[192] = 32;

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoaderFactory.Default.ReadQ6K(reader, data.AsSpan(), 256);

        // Elements 0..15: sc[0]=32, qv=0 → d*32*(0-32) = -1024
        Assert.Equal(-1024.0f, data[0]);
        Assert.Equal(-1024.0f, data[15]);

        // Elements 16..31: sc[1]=0 → d*0*(0-32) = 0
        Assert.Equal(0.0f, data[16]);
        Assert.Equal(0.0f, data[31]);

        // Elements 32..47 (second half, sc starts at 8)
        // sc[0+8] = sc[8] = 0 → value = 0
        // Wait, let me think about the loop structure...

        // nOff=0: qlOff=0, qhOff=0, scOff=0
        //   l=0..31:
        //     is_ = l/16 → 0 for l<16, 1 for l>=16
        //     q1 = 0 (all zeros), sc[0] for l<16, sc[1] for l>=16
        //     q1..q4 = 0 all zeros
        //     
        //     Elements 0(l), 32(l+32), 64(l+64), 96(l+96)
        //
        // nOff=128: qlOff=64, qhOff=32, scOff=8
        //   l=0..31:
        //     Elements 128(l), 160(l+32), 192(l+64), 224(l+96)
        //     sc[8] for l<16, sc[9] for l>=16
        //     sc[10] for l<16, sc[11] for l>=16
        //     sc[12] for l<16, sc[13] for l>=16
        //     sc[14] for l<16, sc[15] for l>=16
        //     
        //   Checks in code:
        //     i2 = b*QK_K + nOff + l + 32 = 160
        //     i2 >= b*QK_K + valid = 256? No, 160 < 256 ✓
        //     i3 = 192, i4 = 224
        //     all within valid(256) ✓

        // nOff=128: sc=sc[8..15] acting on elements 128..255
        // With the way scOff=8, sc starts at sc[8+is_+0]
        // For each nOff, 4 groups of 32 elements:
        //   Elements nOff..nOff+15: sc[scOff + 0] (is_=0)
        //   Elements nOff+32..nOff+47: sc[scOff + 2] (is_=1)
        //   Elements nOff+64..nOff+79: sc[scOff + 4] (is_=0)
        //   Elements nOff+96..nOff+111: sc[scOff + 6] (is_=1)

        // Hmm, wait. Let me trace the nOff=128 case more carefully:
        // nOff=0: elements 0..127. l from 0 to 31.
        //   For l=0 (is_=0): 
        //     i1=0: sc[0] → -1024
        //     i2=32: sc[2] → 0
        //     i3=64: sc[4] → 0
        //     i4=96: sc[6] → 0
        //   For l=16 (is_=1):
        //     i1=16: sc[1] → 0
        //     i2=48: sc[3] → 0
        //     i3=80: sc[5] → 0
        //     i4=112: sc[7] → 0
        //
        // nOff=128: elements 128..255. l from 0 to 31.
        //   scOff=8.
        //   For l=0 (is_=0):
        //     i1=128: sc[8] → 0
        //     i2=160: sc[10] → 0
        //     i3=192: sc[12] → 0
        //     i4=224: sc[14] → 0
        //   For l=16 (is_=1):
        //     i1=144: sc[9] → 0
        //     i2=176: sc[11] → 0
        //     i3=208: sc[15] → 0
        //     i4=240:

        // Wait, the sc indexing access is for each nOff:
        //   l<16 (is_=0): sc[scOff+0], sc[scOff+2], sc[scOff+4], sc[scOff+6]
        //   l>=16 (is_=1): sc[scOff+1], sc[scOff+3], sc[scOff+5], sc[scOff+7]
        // For nOff=0, scOff=0:
        //   l<16: sc[0], sc[2], sc[4], sc[6]
        //   l>=16: sc[1], sc[3], sc[5], sc[7]
        //
        // So for the test:
        // Element 0 (nOff=0, l=0, is_=0): sc[0]=32 → -1024
        // Element 16 (nOff=0, l=16, is_=1): sc[1]=0 → 0
        // Element 32 (nOff=0, l=0, is_=0, q2): sc[2]=0 → 0
        // Element 48 (nOff=0, l=16, is_=1, q2): sc[3]=0 → 0
        // Element 64 (nOff=0, l=0, is_=0, q3): sc[4]=0 → 0
        // Element 96 (nOff=0, l=0, is_=0, q4): sc[6]=0 → 0

        Assert.Equal(-1024.0f, data[0]);
        Assert.Equal(0.0f, data[16]);
        Assert.Equal(0.0f, data[32]);
        Assert.Equal(0.0f, data[64]);
        Assert.Equal(0.0f, data[96]);
    }

    /// <summary>
    /// Verify that an element with qv=1 gets the right value.
    /// Set ql[0] = 0x01 (low nibble = 1), qh[0] & 3 = 0
    /// → q1 = 1 for element 0
    /// sc[0] = 32, d = 1.0
    /// value = 1.0 * 32 * (1 - 32) = -992
    /// </summary>
    [Fact]
    public void TestQ6K_QvNonZero()
    {
        var block = new byte[210];
        block[208] = 0x00; block[209] = 0x3C; // d = 1.0
        block[192] = 32; // sc[0] = 32

        // ql[0] = 0x01 → low nibble = 1
        // qh[0] = 0x00 → high bits = 0
        // q1 = 1 | 0 = 1
        block[0] = 0x01;

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoaderFactory.Default.ReadQ6K(reader, data.AsSpan(), 256);

        // Element 0: d * sc[0] * (1 - 32) = 32 * (-31) = -992
        Assert.Equal(-992.0f, data[0]);
    }

    /// <summary>
    /// Verify q1 with high bits from qh.
    /// ql[0] = 0x0F (low nibble = 15), qh[0] = 0x01 (bits 0-1 = 1)
    /// → q1 = 15 | (1 << 4) = 31
    /// sc[0] = 32, d = 1.0
    /// value = 32 * (31 - 32) = -32
    /// </summary>
    [Fact]
    public void TestQ6K_HighBits()
    {
        var block = new byte[210];
        block[208] = 0x00; block[209] = 0x3C;
        block[192] = 32;

        block[0] = 0x0F;   // ql[0] low nibble = 15
        block[128] = 0x01; // qh[0] bits 0-1 = 1

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoaderFactory.Default.ReadQ6K(reader, data.AsSpan(), 256);

        // q1 = 15 | (1 << 4) = 31
        // value = 32 * (31 - 32) = -32
        Assert.Equal(-32.0f, data[0]);
    }

    /// <summary>
    /// Verify q2 (element 32): ql[32] low nibble, qh[0] bits 2-3.
    /// ql[32] = 0x0F (low nibble = 15), qh[0] = 0x04 (bits 2-3 = 1)
    /// → q2 = 15 | (1 << 4) = 31
    /// sc[2] = 32, d = 1.0
    /// value = 32 * (31 - 32) = -32
    /// But sc[2] = 0 (since only sc[0]=32, rest=0), so value = 0
    /// 
    /// Let me set sc[2] = 32 instead.
    /// </summary>
    [Fact]
    public void TestQ6K_Q2()
    {
        var block = new byte[210];
        block[208] = 0x00; block[209] = 0x3C; // d = 1.0

        // Set all 16 scales to 32
        for (int i = 192; i < 192 + 16; i++)
            block[i] = 32;

        // ql[32] low nibble = 0x0F → 15
        block[32] = 0x0F;
        // qh[0] bits 2-3 = 1 → 0x04
        block[128] = 0x04;

        var ms = new MemoryStream(block);
        var reader = new BinaryReader(ms);
        var data = new float[256];
        GgufLoaderFactory.Default.ReadQ6K(reader, data.AsSpan(), 256);

        // Element 32 (nOff=0, l=0, q2): sc[2]=32, q2=15|(1<<4)=31
        // value = 32 * (31 - 32) = -32
        Assert.Equal(-32.0f, data[32]);

        // Element 0: sc[0]=32, q1 = 0 (no ql/qh set for element 0)
        // value = 32 * (0 - 32) = -1024
        Assert.Equal(-1024.0f, data[0]);
    }
}
