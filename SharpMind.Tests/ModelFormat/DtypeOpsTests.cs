using System.Buffers.Binary;
using SharpMind.Model.Format;
using TensorF = SharpMind.Core.Tensors.Tensor<float>;

namespace SharpMind.Tests.ModelFormat;

public class DtypeOpsTests
{
    [Fact]
    public void ElementSize_F32_Returns4()
    {
        Assert.Equal(4, DtypeOps.ElementSize(Dtype.F32));
    }

    [Fact]
    public void ElementSize_F16_Returns2()
    {
        Assert.Equal(2, DtypeOps.ElementSize(Dtype.F16));
    }

    [Fact]
    public void ElementSize_INT8_Returns1()
    {
        Assert.Equal(1, DtypeOps.ElementSize(Dtype.INT8));
    }

    [Fact]
    public void ElementSize_INT4_Returns1()
    {
        Assert.Equal(1, DtypeOps.ElementSize(Dtype.INT4));
    }

    [Fact]
    public void ConvertToFloat_F32_Roundtrip_Exact()
    {
        var original = new TensorF([4]);
        original.Data[0] = 1.0f;
        original.Data[1] = -2.5f;
        original.Data[2] = 0.0f;
        original.Data[3] = 1000.0f;

        var bytes = DtypeOps.ConvertFromFloat(original, Dtype.F32);
        var result = DtypeOps.ConvertToFloat(bytes, Dtype.F32, 4);

        Assert.Equal(1.0f, result.Data[0], 5);
        Assert.Equal(-2.5f, result.Data[1], 5);
        Assert.Equal(0.0f, result.Data[2], 5);
        Assert.Equal(1000.0f, result.Data[3], 5);

        result.Dispose();
    }

    [Fact]
    public void ConvertToFloat_F16_Roundtrip_Close()
    {
        var original = new TensorF([4]);
        original.Data[0] = 1.0f;
        original.Data[1] = -3.14159f;
        original.Data[2] = 0.0001f;
        original.Data[3] = 65500.0f;

        var bytes = DtypeOps.ConvertFromFloat(original, Dtype.F16);
        var result = DtypeOps.ConvertToFloat(bytes, Dtype.F16, 4);

        Assert.Equal(1.0f, result.Data[0], 0.01f);
        Assert.Equal(-3.14f, result.Data[1], 0.01f);
        Assert.Equal(0.0f, result.Data[2], 0.001f);
        Assert.Equal(65500f, result.Data[3], 100f);

        result.Dispose();
    }

    [Fact]
    public void ConvertToFloat_INT8_Deiquantize()
    {
        var data = new byte[] { 0, 1, 2, 3 };
        var result = DtypeOps.ConvertToFloat(data, Dtype.INT8, 4);

        Assert.Equal(0f, result.Data[0], 1f);
        Assert.Equal(1f, result.Data[1], 1f);
        Assert.Equal(2f, result.Data[2], 1f);
        Assert.Equal(3f, result.Data[3], 1f);

        result.Dispose();
    }

    [Fact]
    public void ConvertToFloat_INT4_Unpack()
    {
        var data = new byte[] { 0x12, 0x34, 0x56, 0x78 };
        var result = DtypeOps.ConvertToFloat(data, Dtype.INT4, 8);

        Assert.Equal(2f, result.Data[0], 1f);
        Assert.Equal(1f, result.Data[1], 1f);
        Assert.Equal(4f, result.Data[2], 1f);
        Assert.Equal(3f, result.Data[3], 1f);
        Assert.Equal(6f, result.Data[4], 1f);
        Assert.Equal(5f, result.Data[5], 1f);
        Assert.Equal(8f, result.Data[6], 1f);
        Assert.Equal(7f, result.Data[7], 1f);

        result.Dispose();
    }

    [Fact]
    public void ConvertToFloat_InvalidDtype_Throws()
    {
        var data = new byte[4];
        Assert.Throws<NotSupportedException>(() => 
            DtypeOps.ConvertToFloat(data, (Dtype)999, 1));
    }

    [Fact]
    public void HalfToFloat_Decode_Subnormals_Are_2ToTheMinus24_Scaled()
    {
        var data = new byte[1023 * 2];
        for (ushort mant = 1; mant <= 1023; mant++)
            BinaryPrimitives.WriteUInt16LittleEndian(data.AsSpan((mant - 1) * 2, 2), mant);

        using var result = DtypeOps.ConvertToFloat(data, Dtype.F16, 1023);
        for (ushort mant = 1; mant <= 1023; mant++)
        {
            float expected = mant * MathF.Pow(2f, -24f);
            Assert.Equal(BitConverter.SingleToUInt32Bits(expected),
                         BitConverter.SingleToUInt32Bits(result.Data[mant - 1]));
        }

        result.Dispose();
    }

    [Fact]
    public void HalfToFloat_Zero_RespectsSignBit()
    {
        using var pos = DtypeOps.ConvertToFloat(new byte[] { 0x00, 0x00 }, Dtype.F16, 1);
        using var neg = DtypeOps.ConvertToFloat(new byte[] { 0x00, 0x80 }, Dtype.F16, 1);

        Assert.Equal(0x00000000u, BitConverter.SingleToUInt32Bits(pos.Data[0]));
        Assert.Equal(0x80000000u, BitConverter.SingleToUInt32Bits(neg.Data[0]));

        pos.Dispose();
        neg.Dispose();
    }

    [Fact]
    public void HalfToFloat_Decode_MinAndMaxSubnormal()
    {
        // 0x0001 -> 2^-24, 0x03FF -> (1023 / 2^24)
        using var min = DtypeOps.ConvertToFloat(new byte[] { 0x01, 0x00 }, Dtype.F16, 1);
        using var max = DtypeOps.ConvertToFloat(new byte[] { 0xFF, 0x03 }, Dtype.F16, 1);

        Assert.Equal(MathF.Pow(2f, -24f), min.Data[0], 2);
        Assert.Equal(1023f * MathF.Pow(2f, -24f), max.Data[0], 2);

        min.Dispose();
        max.Dispose();
    }

    [Fact]
    public void ConvertFromFloat_F16_Subnormals_RoundTrip()
    {
        var original = new TensorF([3]);
        original.Data[0] = MathF.Pow(2f, -24f);
        original.Data[1] = 3f * MathF.Pow(2f, -24f);
        original.Data[2] = 1000f * MathF.Pow(2f, -24f);

        var bytes = DtypeOps.ConvertFromFloat(original, Dtype.F16);
        Assert.Equal(0x0001, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(0, 2)));
        Assert.Equal(0x0003, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(2, 2)));
        Assert.Equal(0x03E8, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(4, 2)));

        using var result = DtypeOps.ConvertToFloat(bytes, Dtype.F16, 3);
        Assert.Equal(original.Data[0], result.Data[0], 2);
        Assert.Equal(original.Data[1], result.Data[1], 2);
        Assert.Equal(original.Data[2], result.Data[2], 2);

        original.Dispose();
        result.Dispose();
    }

    [Fact]
    public void ConvertFromFloat_F16_RoundHalfEven_AtSubnormalBoundary()
    {
        // 2^-25 is the exact midpoint between 0 and the smallest subnormal (2^-24);
        // round-half-to-even must produce 0. Just above the midpoint rounds up to 0x0001.
        var tie = new TensorF([1]);
        tie.Data[0] = MathF.Pow(2f, -25f);
        var tieBytes = DtypeOps.ConvertFromFloat(tie, Dtype.F16);
        Assert.Equal(0x0000, BinaryPrimitives.ReadUInt16LittleEndian(tieBytes.AsSpan(0, 2)));

        var above = new TensorF([1]);
        above.Data[0] = MathF.Pow(2f, -25f) + MathF.Pow(2f, -26f);
        var aboveBytes = DtypeOps.ConvertFromFloat(above, Dtype.F16);
        Assert.Equal(0x0001, BinaryPrimitives.ReadUInt16LittleEndian(aboveBytes.AsSpan(0, 2)));

        tie.Dispose();
        above.Dispose();
    }
}