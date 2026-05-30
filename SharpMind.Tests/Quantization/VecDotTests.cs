using System;
using System.IO;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Quantization;

public class VecDotTests
{
    [Fact]
    public unsafe void TestVecDotQ3K_ValidBlock()
    {
        // 110-byte block: d[2] + hmask[32] + qs[64] + scales[12]
        var block = new byte[110];
        
        // d = 2.0 (F16: 0x4000) - little endian
        block[108] = 0x00;
        block[109] = 0x40;
        
        // scales[0..11] in bit-packed format that decodes to sc[0..15] = 0
        // buf[96..99] low nibble, buf[100..103] low nibble, buf[104..107] = 0xAA gives all p[j] = 0x20, sc = 0x20-32 = 0
        for (int j = 0; j < 4; j++) block[96 + j] = 0x00;  // buf[96..99]
        for (int j = 0; j < 4; j++) block[100 + j] = 0x00;  // buf[100..103]
        for (int j = 0; j < 4; j++) block[104 + j] = 0xAA;  // buf[104..107]

        // qs[0] = 0x00 (all 0s, s2=0)
        block[32] = 0x00;
        
        // hmask[0] = 0x00 (all 0s, hmask byte 0 bit 0 = 0 → actual = s2 - 4 = -4)
        block[0] = 0x00;
        
        var input = new float[] { 1.0f };
        var weights = new byte[110];
        Array.Copy(block, weights, 110);
        
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            // All scales = 32 → dl = d * (32-32) = 0 → val = 0 → sum = 0
            float result = LinearLayer.VecDotQ3K(pInput, pWeights, 0, 1);
            Assert.Equal(0.0f, result, 5);
        }
    }

    [Fact]
    public unsafe void TestVecDotQ4K_ValidBlock()
    {
        // 144-byte block: d[2] + dmin[2] + scales[K_SCALE_SIZE] + qs[128]
        var block = new byte[144];

        // d = 1.0 (0x3C00), min = 0.0 (0x0000)
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;

        // scales[0] = 0x11 → sc=17, m=0 (scales[4]=0x00)
        block[4] = 0x11;

        // qs[0] = 0x11 → val = 1
        block[16] = 0x11;

        // val = d * sc * val - min * m = 17.0 * 1 - 0 = 17.0
        // sum = input[0] * 17.0 = 17.0

        var input = new float[] { 1.0f };
        var weights = new byte[144];
        Array.Copy(block, weights, 144);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = LinearLayer.VecDotQ4K(pInput, pWeights, 0, 1);
            Assert.Equal(17.0f, result, 5);
        }
    }

    [Fact]
    public unsafe void TestVecDotQ5K_ValidBlock()
    {
        // 176-byte block: d[2] + dmin[2] + scales[K_SCALE_SIZE] + qh[32] + qs[128]
        var block = new byte[176];

        // d = 1.0 (0x3C00), min = 0.0 (0x0000)
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;

        // scales[0] = 0x11 → sc=17, m=0
        block[4] = 0x11;

        // qh[0] = 0x00 → high bit disabled
        block[16] = 0x00;

        // qs[0] = 0x11 → val = 1
        block[48] = 0x11;

        // val = d * sc * val - min * m = 17.0 * 1 - 0 = 17.0
        // sum = input[0] * 17.0 = 17.0

        var input = new float[] { 1.0f };
        var weights = new byte[176];
        Array.Copy(block, weights, 176);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = LinearLayer.VecDotQ5K(pInput, pWeights, 0, 1);
            Assert.Equal(17.0f, result, 5);
        }
    }

    [Fact]
    public unsafe void TestVecDotQ8_0_ValidBlock()
    {
        // 34-byte block: d[2] + sbyte[32]
        var block = new byte[34];

        // d = 2.0 (0x4000)
        block[0] = 0x00; block[1] = 0x40;

        // sbyte[0] = 3 → val = 3 * 2.0 = 6.0
        block[2] = 0x03;

        var input = new float[] { 1.0f };
        var weights = new byte[34];
        Array.Copy(block, weights, 34);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = LinearLayer.VecDotQ8_0(pInput, pWeights, 0, 1);
            Assert.Equal(6.0f, result, 5);
        }
    }

    [Fact]
    public unsafe void TestVecDotQ6K_ValidBlock()
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
        // sum = input[0] * 0 = 0

        var input = new float[] { 1.0f };
        var weights = new byte[210];
        Array.Copy(block, weights, 210);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = LinearLayer.VecDotQ6K(pInput, pWeights, 0, 1);
            Assert.Equal(0.0f, result, 5);
        }
    }

    [Fact]
    public unsafe void TestVecDotQ4_0_ValidBlock()
    {
        var block = new byte[18];
        block[0] = 0x00; block[1] = 0x40; // d = 2.0
        block[2] = 0x34; // qs[0] nibble 4
        var input = new float[] { 1.0f };
        var weights = new byte[18];
        Array.Copy(block, weights, 18);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(-8.0f, LinearLayer.VecDotQ4_0(pInput, pWeights, 0, 1), 5); // (4-8)*2.0 = -8.0
    }

    [Fact]
    public unsafe void TestVecDotQ4_1_ValidBlock()
    {
        var block = new byte[20];
        block[0] = 0x00; block[1] = 0x40; // d = 2.0
        block[2] = 0x00; block[3] = 0x00; // m = 0.0
        block[4] = 0x34; // qs[0] nibble 4
        var input = new float[] { 1.0f };
        var weights = new byte[20];
        Array.Copy(block, weights, 20);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(8.0f, LinearLayer.VecDotQ4_1(pInput, pWeights, 0, 1), 5); // 4*2.0+0 = 8.0
    }

    [Fact]
    public unsafe void TestVecDotQ5_0_ValidBlock()
    {
        var block = new byte[22];
        block[0] = 0x00; block[1] = 0x40; // d = 2.0
        block[6] = 0x01; // qs[0] nibble 1
        var input = new float[] { 1.0f };
        var weights = new byte[22];
        Array.Copy(block, weights, 22);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(-30.0f, LinearLayer.VecDotQ5_0(pInput, pWeights, 0, 1), 5); // (1-16)*2.0 = -30.0
    }

    [Fact]
    public unsafe void TestVecDotQ5_1_ValidBlock()
    {
        var block = new byte[24];
        block[0] = 0x00; block[1] = 0x40; // d = 2.0
        block[2] = 0x00; block[3] = 0x00; // m = 0.0
        block[8] = 0x01; // qs[0] nibble 1
        var input = new float[] { 1.0f };
        var weights = new byte[24];
        Array.Copy(block, weights, 24);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(2.0f, LinearLayer.VecDotQ5_1(pInput, pWeights, 0, 1), 5); // 1*2.0+0 = 2.0
    }

    [Fact]
    public unsafe void TestVecDotQ8_1_ValidBlock()
    {
        var block = new byte[36];
        block[0] = 0x00; block[1] = 0x40; // d = 2.0
        block[4] = 0x05; // qs[0] = 5
        var input = new float[] { 1.0f };
        var weights = new byte[36];
        Array.Copy(block, weights, 36);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(10.0f, LinearLayer.VecDotQ8_1(pInput, pWeights, 0, 1), 5); // 5*2.0 = 10.0
    }

    [Fact]
    public unsafe void TestVecDotQ2K_ValidBlock()
    {
        var block = new byte[84];
        block[0] = 0x00; block[1] = 0x3C; // d = 1.0
        block[2] = 0x00; block[3] = 0x00; // dmin = 0.0
        var input = new float[] { 1.0f };
        var weights = new byte[84];
        Array.Copy(block, weights, 84);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(0.0f, LinearLayer.VecDotQ2K(pInput, pWeights, 0, 1), 5);
    }

    [Fact]
    public unsafe void TestVecDotQ8K_ValidBlock()
    {
        var block = new byte[292];
        BitConverter.GetBytes(1.0f).CopyTo(block, 0); // d = 1.0f (F32)
        block[4] = 0x05; // qs[0] = 5
        var input = new float[] { 1.0f };
        var weights = new byte[292];
        Array.Copy(block, weights, 292);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(5.0f, LinearLayer.VecDotQ8K(pInput, pWeights, 0, 1), 5); // 5*1.0 = 5.0
    }
}
