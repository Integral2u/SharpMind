using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Quantization;
using Xunit;

namespace SharpMind.Tests.Quantization;

public class VecDotTests
{
    public static IEnumerable<object[]> AllQuantizationOps()
    {
        yield return new object[] { QuantizationFactory.Create(HardwareTier.Scalar) };
        if (Sse.IsSupported)
        {
            yield return new object[] { QuantizationFactory.Create(HardwareTier.SSE) };
            if (Avx2.IsSupported)
            {
                yield return new object[] { QuantizationFactory.Create(HardwareTier.AVX2) };
                if (Fma.IsSupported)
                    yield return new object[] { QuantizationFactory.Create(HardwareTier.FMA) };
            }
        }
    }

    [Fact]
    public unsafe void VecDotQ3K_MultiBlockAgrees()
    {
        const int blockBytes = 110, qk = 256, nBlocks = 4;
        var rng = new Random(42);
        var weights = new byte[nBlocks * blockBytes];
        rng.NextBytes(weights);
        var input = new float[nBlocks * qk];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() - 0.5);
        int inF = nBlocks * qk;
        var all = new List<QuantizationOps> { QuantizationFactory.Create(HardwareTier.Scalar) };
        if (Sse.IsSupported)
        {
            all.Add(QuantizationFactory.Create(HardwareTier.SSE));
            if (Avx2.IsSupported)
            {
                all.Add(QuantizationFactory.Create(HardwareTier.AVX2));
                if (Fma.IsSupported)
                    all.Add(QuantizationFactory.Create(HardwareTier.FMA));
            }
        }
        var results = new List<float>();
        fixed (float* pIn = input) fixed (byte* pW = weights)
            foreach (var q in all) results.Add(q.VecDotQ3K(pIn, pW, 0, inF));
        var b = (double)results[0];
        for (int i = 1; i < results.Count; i++)
        {
            double err = Math.Abs(b - results[i]);
            Assert.True(err < 1.0 || err / Math.Max(1.0, Math.Abs(b)) < 1e-5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ3K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[110];
        block[108] = 0x00;
        block[109] = 0x40;
        for (int j = 0; j < 4; j++) block[96 + j] = 0x00;
        for (int j = 0; j < 4; j++) block[100 + j] = 0x00;
        for (int j = 0; j < 4; j++) block[104 + j] = 0xAA;
        block[32] = 0x00;
        block[0] = 0x00;

        var input = new float[] { 1.0f };
        var weights = new byte[110];
        Array.Copy(block, weights, 110);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ3K(pInput, pWeights, 0, 1);
            Assert.Equal(0.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[144];
        // dSuper = 1.0 at block[0..1]
        block[0] = 0x00; block[1] = 0x3C;
        // scales[0] = 0x11 → scale=17, scales[4] = 0 → min=0
        block[4] = 0x11;
        // qs[0] low nibble = 1 → v=1
        block[16] = 0x01;

        var input = new float[] { 1.0f };
        var weights = new byte[144];
        Array.Copy(block, weights, 144);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ4K(pInput, pWeights, 0, 1);
            Assert.Equal(17.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ5K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[176];
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;
        block[4] = 0x11;
        block[16] = 0x00;
        block[48] = 0x11;

        var input = new float[] { 1.0f };
        var weights = new byte[176];
        Array.Copy(block, weights, 176);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ5K(pInput, pWeights, 0, 1);
            Assert.Equal(17.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ8_0_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[34];
        block[0] = 0x00; block[1] = 0x40;
        block[2] = 0x03;

        var input = new float[] { 1.0f };
        var weights = new byte[34];
        Array.Copy(block, weights, 34);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ8_0(pInput, pWeights, 0, 1);
            Assert.Equal(6.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ6K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[210];
        block[208] = 0x00;
        block[209] = 0x3C;

        var input = new float[] { 1.0f };
        var weights = new byte[210];
        Array.Copy(block, weights, 210);

        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
        {
            float result = qOps.VecDotQ6K(pInput, pWeights, 0, 1);
            Assert.Equal(0.0f, result, 5);
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_0_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[18];
        block[0] = 0x00; block[1] = 0x40;
        block[2] = 0x34;
        var input = new float[] { 1.0f };
        var weights = new byte[18];
        Array.Copy(block, weights, 18);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(-8.0f, qOps.VecDotQ4_0(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_1_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[20];
        block[0] = 0x00; block[1] = 0x40;
        block[2] = 0x00; block[3] = 0x00;
        block[4] = 0x34;
        var input = new float[] { 1.0f };
        var weights = new byte[20];
        Array.Copy(block, weights, 20);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(8.0f, qOps.VecDotQ4_1(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ5_0_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[22];
        block[0] = 0x00; block[1] = 0x40;
        block[6] = 0x01;
        var input = new float[] { 1.0f };
        var weights = new byte[22];
        Array.Copy(block, weights, 22);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(-30.0f, qOps.VecDotQ5_0(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ5_1_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[24];
        block[0] = 0x00; block[1] = 0x40;
        block[2] = 0x00; block[3] = 0x00;
        block[8] = 0x01;
        var input = new float[] { 1.0f };
        var weights = new byte[24];
        Array.Copy(block, weights, 24);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(2.0f, qOps.VecDotQ5_1(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ8_1_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[36];
        block[0] = 0x00; block[1] = 0x40;
        block[4] = 0x05;
        var input = new float[] { 1.0f };
        var weights = new byte[36];
        Array.Copy(block, weights, 36);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(10.0f, qOps.VecDotQ8_1(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ2K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[84];
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;
        var input = new float[] { 1.0f };
        var weights = new byte[84];
        Array.Copy(block, weights, 84);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(0.0f, qOps.VecDotQ2K(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ8K_ValidBlock(QuantizationOps qOps)
    {
        var block = new byte[292];
        BitConverter.GetBytes(1.0f).CopyTo(block, 0);
        block[4] = 0x05;
        var input = new float[] { 1.0f };
        var weights = new byte[292];
        Array.Copy(block, weights, 292);
        fixed (float* pInput = input)
        fixed (byte* pWeights = weights)
            Assert.Equal(5.0f, qOps.VecDotQ8K(pInput, pWeights, 0, 1), 5);
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_0_FullBlock32_Elements(QuantizationOps qOps)
    {
        const int BLOCK_BYTES = 18;
        const int QK = 32;
        const int N_COLS = 4;
        const int IN_FEATURES = QK;
        var rng = new Random(42);
        var rawWeights = new byte[N_COLS * BLOCK_BYTES];
        var input = new float[IN_FEATURES];

        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        for (int c = 0; c < N_COLS; c++)
        {
            ushort scaleHalf = FloatToHalf(0.5f + rng.NextSingle());
            rawWeights[c * BLOCK_BYTES + 0] = (byte)(scaleHalf & 0xFF);
            rawWeights[c * BLOCK_BYTES + 1] = (byte)(scaleHalf >> 8);
            for (int b = 2; b < BLOCK_BYTES; b++)
                rawWeights[c * BLOCK_BYTES + b] = (byte)rng.Next(256);
        }

        for (int c = 0; c < N_COLS; c++)
        {
            float expected = 0;
            float d = HalfToFloatTest((ushort)(rawWeights[c * BLOCK_BYTES] | ((ushort)rawWeights[c * BLOCK_BYTES + 1] << 8)));
            fixed (byte* qs = &rawWeights[c * BLOCK_BYTES + 2])
            {
                for (int i = 0; i < IN_FEATURES; i++)
                {
                    int q = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
                    expected += input[i] * ((q - 8) * d);
                }
            }

            fixed (float* pIn = input) fixed (byte* pW = rawWeights)
            {
                float result = qOps.VecDotQ4_0(pIn, pW, c, IN_FEATURES);
                Assert.Equal(expected, result, 4);
            }
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_0_MultiBlock64_Elements(QuantizationOps qOps)
    {
        const int BLOCK_BYTES = 18;
        const int QK = 32;
        const int N_COLS = 2;
        const int N_BLOCKS = 2;
        const int IN_FEATURES = N_BLOCKS * QK;
        var rng = new Random(99);
        var rawWeights = new byte[N_COLS * N_BLOCKS * BLOCK_BYTES];
        var input = new float[IN_FEATURES];

        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        for (int c = 0; c < N_COLS; c++)
        {
            for (int b = 0; b < N_BLOCKS; b++)
            {
                int off = c * N_BLOCKS * BLOCK_BYTES + b * BLOCK_BYTES;
                ushort scaleHalf = FloatToHalf(0.3f + rng.NextSingle());
                rawWeights[off + 0] = (byte)(scaleHalf & 0xFF);
                rawWeights[off + 1] = (byte)(scaleHalf >> 8);
                for (int j = 2; j < BLOCK_BYTES; j++)
                    rawWeights[off + j] = (byte)rng.Next(256);
            }
        }

        for (int c = 0; c < N_COLS; c++)
        {
            float expected = 0;
            float d = HalfToFloatTest((ushort)(rawWeights[c * BLOCK_BYTES] | ((ushort)rawWeights[c * BLOCK_BYTES + 1] << 8)));
            for (int b2 = 0; b2 < N_BLOCKS; b2++)
            {
                int off2 = c * N_BLOCKS * BLOCK_BYTES + b2 * BLOCK_BYTES;
                d = HalfToFloatTest((ushort)(rawWeights[off2] | ((ushort)rawWeights[off2 + 1] << 8)));
                fixed (byte* qs = &rawWeights[off2 + 2])
                {
                    for (int i = 0; i < QK; i++)
                    {
                        int q = (qs[i / 2] >> ((i % 2) * 4)) & 0x0F;
                        expected += input[b2 * QK + i] * ((q - 8) * d);
                    }
                }
            }

            fixed (float* pIn = input) fixed (byte* pW = rawWeights)
            {
                float result = qOps.VecDotQ4_0(pIn, pW, c, IN_FEATURES);
                Assert.Equal(expected, result, 3);
            }
        }
    }

    [Theory]
    [MemberData(nameof(AllQuantizationOps))]
    public unsafe void TestVecDotQ4_0_AgreesAcrossTiers()
    {
        const int BLOCK_BYTES = 18;
        const int QK = 32;
        const int N_COLS = 4;
        const int N_BLOCKS = 2;
        const int IN_FEATURES = N_BLOCKS * QK;
        var rng = new Random(7);
        var rawWeights = new byte[N_COLS * N_BLOCKS * BLOCK_BYTES];
        var input = new float[IN_FEATURES];

        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int c = 0; c < N_COLS; c++)
            for (int b = 0; b < N_BLOCKS; b++)
            {
                int off = c * N_BLOCKS * BLOCK_BYTES + b * BLOCK_BYTES;
                rawWeights[off + 0] = 0x00; rawWeights[off + 1] = 0x3C;
                for (int j = 2; j < BLOCK_BYTES; j++)
                    rawWeights[off + j] = (byte)rng.Next(256);
            }

        var all = new List<float>();
        foreach (var tier in Enum.GetValues<HardwareTier>())
        {
            var q = QuantizationFactory.Create(tier);
            fixed (float* pIn = input) fixed (byte* pW = rawWeights)
                all.Add(q.VecDotQ4_0(pIn, pW, 0, IN_FEATURES));
        }

        float baseline = all[0];
        for (int i = 1; i < all.Count; i++)
            Assert.Equal(baseline, all[i], 5);
    }

    private static ushort FloatToHalf(float f)
    {
        unsafe
        {
            uint bits = *(uint*)&f;
            uint sign = (bits >> 16) & 0x8000;
            int exp = (int)((bits >> 23) & 0xFF) - 127 + 15;
            uint mantissa = bits & 0x7FFFFF;
            if (exp <= 0) return (ushort)sign;
            if (exp >= 31) return (ushort)(sign | 0x7C00);
            return (ushort)(sign | ((uint)exp << 10) | (mantissa >> 13));
        }
    }

    private static float HalfToFloatTest(ushort h)
    {
        unsafe
        {
            uint sign = (uint)(h & 0x8000) << 16;
            int exp = (h >> 10) & 0x1F;
            uint mantissa = (uint)(h & 0x3FF) << 13;
            if (exp == 0) { float z = 0; return *(float*)&sign; }
            if (exp == 31) { uint inf = sign | 0x7F800000u; return *(float*)&inf; }
            uint bits = sign | ((uint)(exp - 15 + 127) << 23) | mantissa;
            return *(float*)&bits;
        }
    }
}
