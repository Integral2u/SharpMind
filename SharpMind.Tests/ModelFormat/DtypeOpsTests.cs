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
}