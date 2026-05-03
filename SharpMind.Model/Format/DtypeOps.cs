using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;

namespace SharpMind.Model.Format;

/// <summary>
/// Utilities for dtype conversion between storage and runtime formats.
/// Handles quantization/dequantization for loading external models.
/// </summary>
public static class DtypeOps
{
    /// <summary>Size in bytes per element for each dtype.</summary>
    public static int ElementSize(Dtype dtype) => dtype switch
    {
        Dtype.F32 => 4,
        Dtype.F16 => 2,
        Dtype.BF16 => 2,
        Dtype.INT8 => 1,
        Dtype.INT4 => 1,
        _ => throw new NotSupportedException($"Unsupported dtype: {dtype}")
    };

    /// <summary>Converts stored weights to runtime float32 tensor.</summary>
    public static Tensor<float> ConvertToFloat(ReadOnlySpan<byte> data, Dtype dtype, int count)
    {
        var result = new Tensor<float>([count]);
        
        switch (dtype)
        {
            case Dtype.F32:
                if (data.Length != count * 4)
                    throw new ArgumentException($"F32 data length {data.Length} != {count * 4}");
                var floats = result.Data;
                for (int i = 0; i < count; i++)
                    floats[i] = BitConverter.ToSingle(data[(i * 4)..]);
                break;
                
            case Dtype.F16:
                HalfToFloat(data, result.Data, count);
                break;
                
            case Dtype.BF16:
                Bf16ToFloat(data, result.Data, count);
                break;
                
            case Dtype.INT8:
                Int8ToFloat(data, result.Data, count);
                break;
                
            case Dtype.INT4:
                Int4ToFloat(data, result.Data, count);
                break;
                
            default:
                throw new NotSupportedException($"Unsupported dtype: {dtype}");
        }
        
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HalfToFloat(ReadOnlySpan<byte> src, Span<float> dst, int count)
    {
        for (int i = 0; i < count; i++)
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(src[(i * 2)..]);
            dst[i] = HalfToFloat(bits);
        }
    }
    
    private static float HalfToFloat(ushort bits)
    {
        int sign = (bits >> 15) & 1;
        int exp = (bits >> 10) & 0x1F;
        int mant = bits & 0x3FF;
        
        if (exp == 0)
        {
            if (mant == 0) return -0f;
            // denormal: denormals have effective exponent of -14
            float result = mant / (float)(1 << 14);
            return sign != 0 ? -result : result;
        }
        else if (exp == 31)
        {
            if (mant == 0) return sign != 0 ? float.NegativeInfinity : float.PositiveInfinity;
            return float.NaN;
        }
        
        // Normal number: exponent bias = 15, adjust to float bias = 127
        int floatExp = exp - 15 + 127;
        int floatMant = mant << 13;
        int floatBits = (sign << 31) | (floatExp << 23) | floatMant;
        return BitConverter.Int32BitsToSingle(floatBits);
    }
    
    private static ushort SingleToHalf(float value)
    {
        if (float.IsNaN(value)) return 0x7E0;
        if (float.IsInfinity(value)) return (ushort)(float.IsPositiveInfinity(value) ? 0x3C00 : 0xFC00);
        
        int bits = BitConverter.SingleToInt32Bits(value);
        int sign = (bits >> 16) & 0x8000;
        int exp = ((bits >> 23) & 0xFF) - 127 + 15;
        int mant = (bits >> 13) & 0x3FF;
        
        if (exp <= 0)
        {
            exp = 0;
        }
        else if (exp >= 31)
        {
            return (ushort)(sign | 0x7C00);
        }
        
        return (ushort)(sign | (exp << 10) | mant);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Bf16ToFloat(ReadOnlySpan<byte> src, Span<float> dst, int count)
    {
        for (int i = 0; i < count; i++)
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(src[(i * 2)..]);
            uint floatBits = (uint)bits << 16;
            dst[i] = BitConverter.Int32BitsToSingle((int)floatBits);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Int8ToFloat(ReadOnlySpan<byte> src, Span<float> dst, int count)
    {
        for (int i = 0; i < count; i++)
            dst[i] = (float)src[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Int4ToFloat(ReadOnlySpan<byte> src, Span<float> dst, int count)
    {
        for (int i = 0; i < count; i++)
        {
            byte b = src[i / 2];
            bool lowNibble = (i % 2 == 0);
            int nibble = lowNibble ? (b & 0x0F) : ((b >> 4) & 0x0F);
            // GGUF uses unsigned 4-bit: values 0-15
            dst[i] = nibble;
        }
    }

    /// <summary>Converts runtime float32 to stored dtype.</summary>
    public static byte[] ConvertFromFloat(Tensor<float> src, Dtype dtype)
    {
        int count = src.ElementCount;
        int byteSize = count * ElementSize(dtype);
        var result = new byte[byteSize];

        switch (dtype)
        {
            case Dtype.F32:
                for (int i = 0; i < count; i++)
                    BitConverter.TryWriteBytes(result.AsSpan((i * 4)..), src.Data[i]);
                break;
                
            case Dtype.F16:
                for (int i = 0; i < count; i++)
                {
                    ushort bits = SingleToHalf(src.Data[i]);
                    BinaryPrimitives.WriteUInt16LittleEndian(result.AsSpan((i * 2)..), bits);
                }
                break;
                
            case Dtype.INT8:
                for (int i = 0; i < count; i++)
                    result[i] = (byte)Math.Clamp(src.Data[i], -127, 127);
                break;
                
            default:
                throw new NotSupportedException($"Cannot convert to {dtype} yet");
        }
        
        return result;
    }
}