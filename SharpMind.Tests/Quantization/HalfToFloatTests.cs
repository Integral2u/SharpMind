using System;
using SharpMind.Core.Quantization;
using Xunit;

namespace SharpMind.Tests.Quantization;

public class HalfToFloatTests
{
    [Theory]
    [InlineData(0x0000, 0x00000000, "+0")]
    [InlineData(0x8000, 0x80000000, "-0")]
    [InlineData(0x3C00, 0x3F800000, "1.0")]        // 0b0_01111_0000000000
    [InlineData(0xBC00, 0xBF800000, "-1.0")]       // 0b1_01111_0000000000
    [InlineData(0x3800, 0x3F000000, "0.5")]         // 0b0_01110_0000000000
    [InlineData(0x4000, 0x40000000, "2.0")]         // 0b0_10000_0000000000
    [InlineData(0x7BFF, 0x477FE000, "max normal")]  // 65504
    [InlineData(0x0400, 0x38800000, "min normal")]  // 2^-14
    [InlineData(0x0001, 0x33800000, "min subnormal")] // 2^-24
    [InlineData(0x03FF, 0x387FC000, "max subnormal")]
    [InlineData(0x1955, 0x3B2AA000, "arbitrary normal")]    // exp=6, mant=341
    [InlineData(0x67FF, 0x44FFE000, "large normal")]         // exp=25, mant=1023
    public void HalfToFloat_F16C_ReturnsCorrectBits(ushort half, uint expectedBits, string _)
    {
        float result = QuantizationKernels.HalfToFloat_F16C(half);
        Assert.Equal(expectedBits, BitConverter.SingleToUInt32Bits(result));
    }

    [Fact]
    public void HalfToFloat_F16C_Zero()
    {
        Assert.Equal(0f, QuantizationKernels.HalfToFloat_F16C(0x0000));
        Assert.Equal(-0f, QuantizationKernels.HalfToFloat_F16C(0x8000));
    }

    [Fact]
    public void HalfToFloat_F16C_Infinity()
    {
        Assert.True(float.IsPositiveInfinity(QuantizationKernels.HalfToFloat_F16C(0x7C00)));
        Assert.True(float.IsNegativeInfinity(QuantizationKernels.HalfToFloat_F16C(0xFC00)));
    }

    [Fact]
    public void HalfToFloat_F16C_NaN()
    {
        foreach (var half in new ushort[] { 0x7C01, 0x7C10, 0x7FFF, 0xFC01, 0xFE00, 0xFFFF })
            Assert.True(float.IsNaN(QuantizationKernels.HalfToFloat_F16C(half)));
    }

    [Fact]
    public void HalfToFloat_F16C_BulkOrdering()
    {
        var rng = new Random(12345);
        for (int i = 0; i < 2000; i++)
        {
            ushort a = (ushort)rng.Next(0x0400, 0x7BFF);
            ushort b = (ushort)rng.Next(0x0400, 0x7BFF);
            if (a == b) continue;

            float fa = QuantizationKernels.HalfToFloat_F16C(a);
            float fb = QuantizationKernels.HalfToFloat_F16C(b);
            Assert.Equal(a < b, fa < fb);
        }
    }

    [Fact]
    public void HalfToFloat_F16C_SubnormalRoundTripBoundary()
    {
        for (ushort mant = 1; mant < 1024; mant++)
        {
            ushort half = (ushort)(0x0000 | mant);
            float result = QuantizationKernels.HalfToFloat_F16C(half);
            Assert.True(result > 0f);
            Assert.True(float.IsFinite(result));
        }
    }
}
