using System.Runtime.CompilerServices;

namespace SharpMind.Core.Quantization;

/// <summary>
/// Converts full-float (F32) weight buffers into the raw byte layout a
/// SharpMind loader / quantized kernel can consume, byte-for-byte compatible
/// with what GGUF stores for the same dtype.
///
/// Supported targets: <see cref="QuantDType.F32"/> (passthrough),
/// <see cref="QuantDType.F16"/> (always safe), and the block formats
/// <see cref="QuantDType.Q8_0"/> / <see cref="QuantDType.Q4_0"/>.
///
/// The block formats are only layout-correct when every tensor dimension is a
/// multiple of the block size (32): the quantized forward kernels group
/// weights per output column, the byte-count helpers group per row, and the
/// two only agree when the whole flattened buffer splits into aligned 32-blocks.
/// <see cref="Quantize"/> enforces this and throws otherwise — the safe
/// fallback for odd-sized tensors is F32 or F16.
/// </summary>
public static partial class TensorQuantizer
{
    private const int Qk = 32;

    /// <summary>True when <paramref name="dtype"/> can be produced by <see cref="Quantize"/>.</summary>
    public static bool IsSupportedTarget(QuantDType dtype) => dtype switch
    {
        QuantDType.F32 or QuantDType.F16 or QuantDType.Q8_0 or QuantDType.Q4_0
            or QuantDType.Q2_K or QuantDType.Q2_K_S
            or QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L
            or QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M
            or QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M
            or QuantDType.Q6_K or QuantDType.Q6_K_S or QuantDType.Q8_K => true,
        _ => false,
    };

    /// <summary>
    /// Quantizes a flat float buffer (the GGUF-layout data for the tensor,
    /// i.e. already row-major over the stored shape) into the raw bytes for
    /// <paramref name="dtype"/>.
    /// </summary>
    public static byte[] Quantize(ReadOnlySpan<float> values, int[] shape, QuantDType dtype)
    {
        return dtype switch
        {
            QuantDType.F32 => WriteF32(values),
            QuantDType.F16 => WriteF16(values),
            QuantDType.Q8_0 => WriteBlock(values, shape, blockBytes: 34, isQ4: false),
            QuantDType.Q4_0 => WriteBlock(values, shape, blockBytes: 18, isQ4: true),
            QuantDType.Q2_K or QuantDType.Q2_K_S => WriteKQ2(values),
            QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L => WriteKQ3(values),
            QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M => WriteKQ4(values),
            QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M => WriteKQ5(values),
            QuantDType.Q6_K or QuantDType.Q6_K_S => WriteKQ6(values),
            QuantDType.Q8_K => WriteKQ8(values),
            _ => throw new NotSupportedException(
                $"Quantization to {dtype} is not supported. Use F32, F16, Q8_0, Q4_0 or a K-quant (Q2_K..Q8_K).")
        };
    }

    private static byte[] WriteF32(ReadOnlySpan<float> values)
    {
        var result = new byte[checked(values.Length * 4)];
        Buffer.BlockCopy(values.ToArray(), 0, result, 0, result.Length);
        return result;
    }

    private static byte[] WriteF16(ReadOnlySpan<float> values)
    {
        var result = new byte[checked(values.Length * 2)];
        for (int i = 0; i < values.Length; i++)
        {
            ushort half = QuantizationKernels.FloatToHalf_F16C(values[i]);
            result[i * 2] = (byte)half;
            result[i * 2 + 1] = (byte)(half >> 8);
        }
        return result;
    }

    private static byte[] WriteBlock(ReadOnlySpan<float> values, int[] shape, int blockBytes, bool isQ4)
    {
        foreach (int d in shape)
        {
            if (d % Qk != 0)
                throw new InvalidOperationException(
                    $"Cannot block-quantize tensor of shape [{string.Join(", ", shape)}]: " +
                    $"every dimension must be a multiple of {Qk} for the Q4_0/Q8_0 layout. " +
                    "Use F16 or keep F32 for this tensor.");
        }
        if (values.Length % Qk != 0)
            throw new InvalidOperationException("Float buffer length must be a multiple of 32 for block quantization.");

        int nBlocks = values.Length / Qk;
        var result = new byte[checked(nBlocks * blockBytes)];

        for (int b = 0; b < nBlocks; b++)
        {
            int blockStart = b * Qk;
            float amax = 0f;
            for (int j = 0; j < Qk; j++)
            {
                float a = MathF.Abs(values[blockStart + j]);
                if (a > amax) amax = a;
            }

            int outBlock = b * blockBytes;
            if (amax == 0f)
            {
                WriteHalf(result, outBlock, 0);
                continue;
            }

            float d = isQ4 ? amax / 7f : amax / 127f;
            WriteHalf(result, outBlock, d);

            int dataStart = outBlock + 2;
            if (isQ4)
            {
                for (int j = 0; j < Qk; j++)
                {
                    int q = (int)MathF.Round(values[blockStart + j] / d, MidpointRounding.AwayFromZero) + 8;
                    if (q < 0) q = 0;
                    if (q > 15) q = 15;
                    int byteIdx = dataStart + (j < Qk / 2 ? j : j - Qk / 2);
                    if (j < Qk / 2)
                        result[byteIdx] = (byte)((result[byteIdx] & 0xF0) | (byte)q);
                    else
                        result[byteIdx] = (byte)((result[byteIdx] & 0x0F) | (byte)(q << 4));
                }
            }
            else
            {
                for (int j = 0; j < Qk; j++)
                {
                    float scaled = values[blockStart + j] / d;
                    int q = (int)MathF.Round(scaled, MidpointRounding.AwayFromZero);
                    if (q < sbyte.MinValue) q = sbyte.MinValue;
                    if (q > sbyte.MaxValue) q = sbyte.MaxValue;
                    result[dataStart + j] = unchecked((byte)(sbyte)q);
                }
            }
        }
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void WriteHalf(byte[] dest, int offset, float value)
    {
        ushort half = QuantizationKernels.FloatToHalf_F16C(value);
        dest[offset] = (byte)half;
        dest[offset + 1] = (byte)(half >> 8);
    }
}
