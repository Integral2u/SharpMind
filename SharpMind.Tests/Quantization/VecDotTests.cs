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
    public void RunConsistencyDiagnostic()
    {
        QuantizationDiagnostic.RunDiagnostics();
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
        block[0] = 0x00; block[1] = 0x3C;
        block[2] = 0x00; block[3] = 0x00;
        block[4] = 0x11;
        block[16] = 0x11;

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
            Assert.Equal(-30.0f, qOps.VecDotQ5_1(pInput, pWeights, 0, 1), 5);
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
}
