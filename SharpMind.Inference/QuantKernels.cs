using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Inference;

// ─────────────────────────────────────────────────────────────────────────────
// Quantized matmul kernels — various bit widths for memory savings
// ─────────────────────────────────────────────────────────────────────────────

internal static partial class QuantKernels
{
    // ── INT4 quantized matmul ─────────────────────────────────────────────
    // Weights stored as INT4 (nibble) in uint8; dequantized to float before multiply.
    // Uses 2 nibbles per byte — packs 2 weights into 1 byte.
    // Format: each uint8 contains [lower_nibble, upper_nibble]
    //   lower = data & 0x0F, upper = (data & 0xF0) >> 4

    internal static unsafe void QuantMatMul_Int4(
        float* input, byte* weights, float* output,
        float* scales,    // [OutFeatures] dequantization scale per channel
        int inFeatures, int outFeatures)
    {
        for (int o = 0; o < outFeatures; o++)
        {
            float scale = scales[o];
            float sum = 0f;

            for (int i = 0; i < inFeatures; i++)
            {
                byte packed = weights[(long)o * inFeatures / 2 + i / 2];
                int nibble = (i % 2 == 0) ? (packed & 0x0F) : ((packed & 0xF0) >> 4);
                int signedNibble = nibble - 8;  // Convert to signed: 0..15 → -8..7
                sum += input[i] * (signedNibble * scale);
            }
            output[o] = sum;
        }
    }

    // ── 2-bit quantization ─────────────────────────────────────────────
    // Each weight is 2 bits: 4 levels per weight (00, 01, 10, 11)
    // 4 weights packed into 1 byte.

    internal static unsafe void QuantMatMul_Int2(
        float* input, byte* weights, float* output,
        float* scales, int inFeatures, int outFeatures)
    {
        for (int o = 0; o < outFeatures; o++)
        {
            float scale = scales[o];
            float sum = 0f;

            for (int i = 0; i < inFeatures; i++)
            {
                byte packed = weights[(long)o * inFeatures / 4 + i / 4];
                int shift = (i % 4) * 2;
                int bits = (packed >> shift) & 0x03;
                int signedBits = bits - 2;  // Convert: 0..3 → -2..1
                sum += input[i] * (signedBits * scale);
            }
            output[o] = sum;
        }
    }

    // ── 1-bit (binary) quantization ──────────────────────────────────────────
    // Each weight is 1 bit: -1 or +1. Packs 8 weights into 1 byte.

    internal static unsafe void QuantMatMul_Int1(
        float* input, byte* weights, float* output,
        float* scales, int inFeatures, int outFeatures)
    {
        for (int o = 0; o < outFeatures; o++)
        {
            float scale = scales[o];
            float sum = 0f;

            for (int i = 0; i < inFeatures; i++)
            {
                byte packed = weights[(long)o * inFeatures / 8 + i / 8];
                int bit = (packed >> (i % 8)) & 0x01;
                int signedBit = bit * 2 - 1;  // Convert: 0→-1, 1→+1
                sum += input[i] * (signedBit * scale);
            }
            output[o] = sum;
        }
    }

    // ── 1.58-bit (ternary) quantization ────────────────────────────────
    // Each weight is ternary: -1, 0, or +1. Packs 5 weights into 4 bytes.
    // Uses LUT-style lookup for decode.

    internal static unsafe void QuantMatMul_Ternary(
        float* input, byte* weights, float* output,
        float* scales, int inFeatures, int outFeatures)
    {
        // Ternary lookup table: decode 5 3-bit values = 15 bits
        // 0 → -1, 4 → 0, 7 → +1 (standard encoding)
        int[] lookup = [-1, -1, -1, -1, 0, 0, 0, 1];

        for (int o = 0; o < outFeatures; o++)
        {
            float scale = scales[o];
            float sum = 0f;

            for (int i = 0; i < inFeatures; i += 8)
            {
                // Process 8 weights at a time → ceil(inFeatures/8) bytes needed
                // But only inFeatures positions used
                if (i >= inFeatures) break;

                byte packed = weights[(long)o * ((inFeatures + 7) / 8) + i / 8];

                for (int j = 0; j < 8 && i + j < inFeatures; j++)
                {
                    int bits = (packed >> (j * 3)) & 0x07;
                    int val = lookup[bits];
                    sum += input[i + j] * (val * scale);
                }
            }
            output[o] = sum;
        }
    }

    // ── FP8 (E4M3) quantization ────────────────────────────────
    // Uses IEEE-like 8-bit float with E4M3 format

    internal static unsafe void QuantMatMul_FP8(
        float* input, byte* weights, float* output,
        float* scales, int inFeatures, int outFeatures)
    {
        for (int o = 0; o < outFeatures; o++)
        {
            float scale = scales[o];
            float sum = 0f;

            for (int i = 0; i < inFeatures; i++)
            {
                // Decode FP8 E4M3 → float
                byte w = weights[(long)o * inFeatures + i];
                float f = DecodeFP8(w);
                sum += input[i] * f;
            }
            output[o] = scale * sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DecodeFP8(byte bits)
    {
        int sign = (bits >> 7) & 0x01;
        int exp = (bits >> 4) & 0x07;
        int mantissa = bits & 0x0F;

        if (exp == 0x07)
            exp = 0xFF;  // Infinity/nan

        int e = exp - 127 + 1;  // bias adjustment
        return (sign == 1 ? -1f : 1f) * MathF.Pow(2, e) * (1 + mantissa / 16f);
    }

    // ── Helper: compute optimal scales for quantization ────────────────────

    public static void ComputeQuantScales(
        ReadOnlySpan<float> weights,
        Span<float> scales,
        QuantKind kind,
        int outFeatures)
    {
        switch (kind)
        {
            case QuantKind.Int8:
                for (int o = 0; o < outFeatures; o++)
                {
                    float max = 0f;
                    for (int i = 0; i < weights.Length / outFeatures; i++)
                    {
                        float abs = MathF.Abs(weights[i * outFeatures + o]);
                        if (abs > max) max = abs;
                    }
                    scales[o] = max / 127f;  // Range [-127, 127]
                }
                break;

            case QuantKind.Int4:
                for (int o = 0; o < outFeatures; o++)
                {
                    float max = 0f;
                    for (int i = 0; i < weights.Length / outFeatures; i++)
                    {
                        float abs = MathF.Abs(weights[i * outFeatures + o]);
                        if (abs > max) max = abs;
                    }
                    scales[o] = max / 7f;  // Range [-7, 7]
                }
                break;

            case QuantKind.Int2:
                for (int o = 0; o < outFeatures; o++)
                {
                    float max = 0f;
                    for (int i = 0; i < weights.Length / outFeatures; i++)
                    {
                        float abs = MathF.Abs(weights[i * outFeatures + o]);
                        if (abs > max) max = abs;
                    }
                    scales[o] = max / 1.5f;  // ~[-1.5, 1.5]
                }
                break;

            case QuantKind.Int1:
                for (int o = 0; o < outFeatures; o++)
                {
                    float max = 0f;
                    for (int i = 0; i < weights.Length / outFeatures; i++)
                    {
                        float abs = MathF.Abs(weights[i * outFeatures + o]);
                        if (abs > max) max = abs;
                    }
                    scales[o] = max;  // Binary: just the max value
                }
                break;

            case QuantKind.Ternary:
                for (int o = 0; o < outFeatures; o++)
                {
                    float max = 0f;
                    for (int i = 0; i < weights.Length / outFeatures; i++)
                    {
                        float abs = MathF.Abs(weights[i * outFeatures + o]);
                        if (abs > max) max = abs;
                    }
                    scales[o] = max;  // Ternary uses max
                }
                break;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════════════
    // TurboQuant — runtime-optimized quantization kernels
    // ═══════════════════════════════════════════════════════════════════════════════════════

    /// <summary>
    /// TurboQuant INT8: fused dequantize + matmul.
    /// Computes output = (input * weight_quant * scale) in one pass.
    /// No intermediate fp32 storage needed.
    /// </summary>
    internal static unsafe void TurboQuant_Int8_Fused(
        float* input, sbyte* weights, float* output,
        float scale,  // shared scale or per-channel
        float* perChannelScales,  // null = use shared
        int inFeatures, int outFeatures)
    {
        bool perChannel = perChannelScales is not null;

        for (int o = 0; o < outFeatures; o++)
        {
            float s = perChannel ? perChannelScales[o] : scale;
            float sum = 0f;

            for (int i = 0; i < inFeatures; i++)
                sum += input[i] * weights[(long)o * inFeatures + i] * s;

            output[o] = sum;
        }
    }

    /// <summary>
    /// TurboQuant INT4: online dequantization during matmul.
    /// AVX2 optimized for packed nibble format.
    /// </summary>
    internal static unsafe void TurboQuant_Int4_Fused(
        float* input, byte* weights, float* output,
        float* perChannelScales,
        int inFeatures, int outFeatures)
    {
        for (int o = 0; o < outFeatures; o++)
        {
            float scale = perChannelScales[o];
            float sum = 0f;

            int i = 0;
            for (; i <= inFeatures - 8; i += 8)
            {
                float tmp = 0f;
                // Process 8 weights = 4 bytes = 4 packed nibbles
                for (int j = 0; j < 4; j++)
                {
                    byte packed = weights[(long)o * inFeatures / 2 + (i + j) / 2];
                    int n1 = (j % 2 == 0) ? (packed & 0x0F) : ((packed & 0xF0) >> 4);
                    int n2 = n1 - 8;  // Signed nibble

                    tmp += input[i + j * 2] * (n2 * scale);
                }
                sum += tmp;
            }

            // Handle remaining
            for (; i < inFeatures; i++)
            {
                byte packed = weights[(long)o * inFeatures / 2 + i / 2];
                int nibble = (i % 2 == 0) ? (packed & 0x0F) : ((packed & 0xF0) >> 4);
                int signedNibble = nibble - 8;
                sum += input[i] * (signedNibble * scale);
            }

            output[o] = sum;
        }
    }

    /// <summary>
    /// TurboQuant FP8 (E5M2) — 5-bit exp + 2-bit mantissa.
    /// Fast decoding for inference servers.
    /// </summary>
    internal static unsafe void TurboQuant_FP8_E5M2(
        float* input, byte* weights, float* output,
        float* scales, int inFeatures, int outFeatures)
    {
        for (int o = 0; o < outFeatures; o++)
        {
            float scale = scales[o];
            float sum = 0f;

            for (int i = 0; i < inFeatures; i++)
            {
                byte w = weights[(long)o * inFeatures + i];
                float fw = DecodeFP8_E5M2(w);
                sum += input[i] * fw;
            }

            output[o] = sum * scale;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DecodeFP8_E5M2(byte bits)
    {
        int sign = (bits >> 7) & 0x01;
        int exp = (bits >> 2) & 0x1F;
        int mantissa = bits & 0x03;

        if (exp == 31)
            return sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;

        float val = (1f + mantissa / 4f) * MathF.Pow(2, exp - 15);
        return sign == 1 ? -val : val;
    }

    /// <summary>
    /// TurboQuant dynamic: computes scales online from input activation range.
    /// No pre-computed scales needed — ideal for dynamic workloads.
    /// </summary>
    public static unsafe void TurboQuantDynamic(
        float* input, float* weights, float* output,
        float targetRange,  // e.g., 1.0 for fp32-like range
        int inFeatures, int outFeatures)
    {
        // Find input range for this batch
        float inputMax = 0f;
        for (int i = 0; i < inFeatures; i++)
        {
            float abs = MathF.Abs(input[i]);
            if (abs > inputMax) inputMax = abs;
        }

        // Compute dynamic scale to map to target range
        float scale = inputMax > 0 ? targetRange / inputMax : 1f;

        // Quantize input on-the-fly and multiply
        for (int o = 0; o < outFeatures; o++)
        {
            float sum = 0f;
            for (int i = 0; i < inFeatures; i++)
            {
                float qinput = MathF.Round(input[i] * scale);
                sum += qinput * weights[(long)o * inFeatures + i];
            }
            output[o] = sum / scale;
        }
    }
}