using System.Runtime.CompilerServices;

namespace SharpMind.Core.Quantization;

/// <summary>
/// K-quant encoder half of <see cref="TensorQuantizer"/>. Produces the raw
/// byte layouts for Q2_K..Q8_K (QK_K = 256 element blocks), byte-for-byte
/// compatible with what the SharpMind decode / VecDot kernels consume.
///
/// Layouts (all offsets within a block):
///   Q2_K (84B):  scales[16] + qs[64] + d[2] + dmin[2]
///   Q3_K (110B): hmask[32] + qs[64] + scales[12] + d[2]
///   Q4_K (144B): d[2] + dmin[2] + scales[12] + qs[128]
///   Q5_K (176B): d[2] + dmin[2] + scales[12] + qh[32] + qs[128]
///   Q6_K (210B): ql[128] + qh[64] + scales[16] + d[2]
///   Q8_K (292B): d[4 (f32)] + qs[256 (i8)]
///
/// The encoders use a simple per-group affine fit. They are layout-correct
/// (round-trippable through the existing readers), not bit-exact vs llama.cpp.
/// </summary>
public static partial class TensorQuantizer
{
    private const int QK_K = 256;

    private static void EnsureKQuantBlockAligned(ReadOnlySpan<float> values)
    {
        if (values.Length % QK_K != 0)
            throw new InvalidOperationException(
                $"Cannot K-quantize buffer of {values.Length} floats: the K-quant formats require " +
                $"a flattened length that is a multiple of {QK_K}. Use F16 or keep F32.");
    }

    private static byte[] WriteKQ2(ReadOnlySpan<float> values)
    {
        EnsureKQuantBlockAligned(values);
        int nBlocks = values.Length / QK_K;
        var result = new byte[checked(nBlocks * 84)];
        int vOff = 0;

        Span<int> codes = stackalloc int[16];
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 84;
            float amax = BlockAbsMax(values, vOff);
            if (amax == 0f)
            {
                vOff += QK_K;
                continue;
            }

            float step = amax / 15f;
            WriteHalf(result, o + 80, step);
            WriteHalf(result, o + 82, step);

            for (int g = 0; g < 16; g++)
            {
                int gOff = vOff + g * 16;
                (float gmn, float gmx) = GroupMinMax(values, gOff, group: 16);
                float range = gmx - gmn;

                int s, m;
                if (range <= 0f)
                {
                    s = 0;
                    m = Clamp((int)MathF.Round(-gmn / step, MidpointRounding.AwayFromZero), 0, 15);
                    codes.Clear();
                }
                else
                {
                    s = Clamp((int)MathF.Round(range / (3f * step), MidpointRounding.AwayFromZero), 1, 15);
                    m = Clamp((int)MathF.Round(-gmn / step, MidpointRounding.AwayFromZero), 0, 15);
                    float sc = s * step;
                    for (int j = 0; j < 16; j++)
                        codes[j] = Clamp((int)MathF.Round((values[gOff + j] + m * step) / sc, MidpointRounding.AwayFromZero), 0, 3);
                }

                result[o + g] = (byte)((s & 0x0F) | ((m & 0x0F) << 4));
                for (int j = 0; j < 16; j++)
                {
                    int i = g * 16 + j;
                    int qsByte = (i / 128) * 32 + (i % 32);
                    int qsShift = ((i % 128) / 32) * 2;
                    result[o + 16 + qsByte] |= (byte)(codes[j] << qsShift);
                }
            }
            vOff += QK_K;
        }
        return result;
    }

    private static byte[] WriteKQ3(ReadOnlySpan<float> values)
    {
        EnsureKQuantBlockAligned(values);
        int nBlocks = values.Length / QK_K;
        var result = new byte[checked(nBlocks * 110)];
        int vOff = 0;

        Span<int> scaleRaw = stackalloc int[16];
        Span<int> bestCodes = stackalloc int[16];
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 110;
            float amax = BlockAbsMax(values, vOff);
            if (amax == 0f)
            {
                vOff += QK_K;
                continue;
            }

            float dAll = amax / 128f;
            WriteHalf(result, o + 108, dAll);

            for (int g = 0; g < 16; g++)
            {
                int gOff = vOff + g * 16;
                (float gmn, float gmx) = GroupMinMax(values, gOff, group: 16);
                float maxA = MathF.Max(MathF.Abs(gmn), MathF.Abs(gmx));
                bestCodes.Clear();

                int gBest = 1;
                if (maxA > 0f)
                {
                    int sGuess = Clamp((int)MathF.Round(maxA / (4f * dAll), MidpointRounding.AwayFromZero), 1, 31);
                    long bestErr = long.MaxValue;
                    for (int delta = -2; delta <= 2; delta++)
                    {
                        int mag = Clamp(sGuess + delta, 1, 31);
                        foreach (int cand in new[] { mag, -mag })
                        {
                            float sc = dAll * cand;
                            long err = 0;
                            for (int j = 0; j < 16; j++)
                            {
                                float x = values[gOff + j];
                                int q = Clamp((int)MathF.Round(x / sc, MidpointRounding.AwayFromZero), -4, 3);
                                float d = x - sc * q;
                                err += (long)(d * d);
                            }
                            if (err < bestErr)
                            {
                                bestErr = err;
                                gBest = cand;
                            }
                        }
                    }
                    float scQ3 = dAll * gBest;
                    for (int j = 0; j < 16; j++)
                        bestCodes[j] = Clamp((int)MathF.Round(values[gOff + j] / scQ3, MidpointRounding.AwayFromZero), -4, 3);
                }

                scaleRaw[g] = Clamp(gBest + 32, 0, 63);
                for (int j = 0; j < 16; j++)
                {
                    int i = g * 16 + j;
                    int code = bestCodes[j];
                    bool positive = code >= 0;
                    int low2 = positive ? code : code + 4;

                    int qsByte = (i / 128) * 32 + (i % 32);
                    int qsShift = ((i % 128) / 32) * 2;
                    result[o + 32 + qsByte] |= (byte)((low2 & 3) << qsShift);
                    if (positive)
                        result[o + i % 32] |= (byte)(1 << (i / 32));
                }
            }

            PackSixBitTwelve(scaleRaw, result, o + 96);
            vOff += QK_K;
        }
        return result;
    }

    private static byte[] WriteKQ4(ReadOnlySpan<float> values)
    {
        EnsureKQuantBlockAligned(values);
        int nBlocks = values.Length / QK_K;
        var result = new byte[checked(nBlocks * 144)];
        int vOff = 0;

        Span<int> sc = stackalloc int[8];
        Span<int> mn = stackalloc int[8];
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 144;
            float amax = BlockAbsMax(values, vOff);
            if (amax == 0f)
            {
                vOff += QK_K;
                continue;
            }

            float step = amax / 63f;
            WriteHalf(result, o, step);
            WriteHalf(result, o + 2, step);

            for (int sub = 0; sub < 8; sub++)
            {
                int gOff = vOff + sub * 32;
                (float gmn, float gmx) = GroupMinMax(values, gOff, group: 32);
                float range = gmx - gmn;

                if (range <= 0f)
                {
                    sc[sub] = 0;
                    mn[sub] = Clamp((int)MathF.Round(-gmn / step, MidpointRounding.AwayFromZero), 0, 63);
                    continue;
                }

                sc[sub] = Clamp((int)MathF.Round(range / (15f * step), MidpointRounding.AwayFromZero), 1, 63);
                mn[sub] = Clamp((int)MathF.Round(-gmn / step, MidpointRounding.AwayFromZero), 0, 63);
                float scF = sc[sub] * step;
                for (int j = 0; j < 32; j++)
                {
                    int code = Clamp((int)MathF.Round((values[gOff + j] + mn[sub] * step) / scF, MidpointRounding.AwayFromZero), 0, 15);
                    int i = sub * 32 + j;
                    int qsByte = (i / 64) * 32 + (i % 32);
                    int qsShift = ((i % 64) / 32) * 4;
                    result[o + 16 + qsByte] |= (byte)(code << qsShift);
                }
            }
            PackScaleMinK4(sc, mn, result, o + 4);
            vOff += QK_K;
        }
        return result;
    }

    private static byte[] WriteKQ5(ReadOnlySpan<float> values)
    {
        EnsureKQuantBlockAligned(values);
        int nBlocks = values.Length / QK_K;
        var result = new byte[checked(nBlocks * 176)];
        int vOff = 0;

        Span<int> sc5 = stackalloc int[8];
        Span<int> mn5 = stackalloc int[8];
        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 176;
            float amax = BlockAbsMax(values, vOff);
            if (amax == 0f)
            {
                vOff += QK_K;
                continue;
            }

            float step = amax / 63f;
            WriteHalf(result, o, step);
            WriteHalf(result, o + 2, step);

            for (int sub = 0; sub < 8; sub++)
            {
                int gOff = vOff + sub * 32;
                (float gmn, float gmx) = GroupMinMax(values, gOff, group: 32);
                float range = gmx - gmn;

                int scVal, mnVal;
                if (range <= 0f)
                {
                    scVal = 0;
                    mnVal = Clamp((int)MathF.Round(-gmn / step, MidpointRounding.AwayFromZero), 0, 63);
                }
                else
                {
                    scVal = Clamp((int)MathF.Round(range / (31f * step), MidpointRounding.AwayFromZero), 1, 63);
                    mnVal = Clamp((int)MathF.Round(-gmn / step, MidpointRounding.AwayFromZero), 0, 63);
                }
                sc5[sub] = scVal;
                mn5[sub] = mnVal;

                float scF = scVal * step;
                for (int j = 0; j < 32; j++)
                {
                    int q5 = range <= 0f
                        ? 0
                        : Clamp((int)MathF.Round((values[gOff + j] + mnVal * step) / scF, MidpointRounding.AwayFromZero), 0, 31);

                    int i = sub * 32 + j;
                    int idx32 = i % 32;
                    int group64 = i / 64;
                    int half = (i % 64) / 32;
                    int bitPos = group64 * 2 + half;

                    int qsIdx = o + 48 + group64 * 32 + idx32;
                    if (half == 0)
                        result[qsIdx] = (byte)((result[qsIdx] & 0xF0) | (q5 & 0x0F));
                    else
                        result[qsIdx] = (byte)((result[qsIdx] & 0x0F) | ((q5 & 0x0F) << 4));

                    int hb = (q5 & 16) >> 4;
                    int qhIdx = o + 16 + idx32;
                    if (hb != 0)
                        result[qhIdx] |= (byte)(1 << bitPos);
                    else
                        result[qhIdx] &= unchecked((byte)~(1 << bitPos));
                }
            }
            PackScaleMinK4(sc5, mn5, result, o + 4);
            vOff += QK_K;
        }
        return result;
    }

    private static byte[] WriteKQ6(ReadOnlySpan<float> values)
    {
        EnsureKQuantBlockAligned(values);
        int nBlocks = values.Length / QK_K;
        var result = new byte[checked(nBlocks * 210)];
        int vOff = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 210;
            float amax = BlockAbsMax(values, vOff);
            if (amax == 0f)
            {
                vOff += QK_K;
                continue;
            }

            float d = amax / 4064f;
            WriteHalf(result, o + 208, d);

            for (int g = 0; g < 16; g++)
            {
                int gOff = vOff + g * 16;
                float gmx = 0f;
                for (int j = 0; j < 16; j++)
                {
                    float a = MathF.Abs(values[gOff + j]);
                    if (a > gmx) gmx = a;
                }

                int scale = gmx == 0f
                    ? 0
                    : Clamp((int)MathF.Round(gmx / (32f * d), MidpointRounding.AwayFromZero), 1, 127);
                result[o + 192 + g] = unchecked((byte)(sbyte)scale);

                float sc = d * scale;
                for (int j = 0; j < 16; j++)
                {
                    int i = g * 16 + j;
                    int code = gmx == 0f
                        ? 32
                        : Clamp((int)MathF.Round(values[gOff + j] / sc, MidpointRounding.AwayFromZero) + 32, 0, 63);
                    WriteQ6Code(result, o, i, code);
                }
            }
            vOff += QK_K;
        }
        return result;
    }

    private static void WriteQ6Code(byte[] result, int o, int i, int code)
    {
        int nOff = (i / 128) * 128;
        int local = i - nOff;
        int l = local % 32;
        int col = local / 32;
        int qlOff = nOff == 0 ? 0 : 64;
        int qhOff = nOff == 0 ? 0 : 32;

        int qlIdx = o + qlOff + (col == 1 || col == 3 ? l + 32 : l);
        int curr = result[qlIdx];
        if (col == 2 || col == 3)
            result[qlIdx] = (byte)((curr & 0x0F) | ((code & 0x0F) << 4));
        else
            result[qlIdx] = (byte)((curr & 0xF0) | (code & 0x0F));

        int qhIdx = o + 128 + qhOff + l;
        int high = (code >> 4) & 0x03;
        int shift = col * 2;
        int cur = result[qhIdx];
        cur &= unchecked((byte)~(0x03 << shift));
        cur |= (byte)(high << shift);
        result[qhIdx] = (byte)cur;
    }

    private static byte[] WriteKQ8(ReadOnlySpan<float> values)
    {
        EnsureKQuantBlockAligned(values);
        int nBlocks = values.Length / QK_K;
        var result = new byte[checked(nBlocks * 292)];
        int vOff = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            int o = b * 292;
            float amax = BlockAbsMax(values, vOff);
            if (amax == 0f)
            {
                vOff += QK_K;
                continue;
            }

            float d = amax / 127f;
            BitConverter.TryWriteBytes(result.AsSpan(o), d);

            for (int j = 0; j < QK_K; j++)
            {
                int q = Clamp((int)MathF.Round(values[vOff + j] / d, MidpointRounding.AwayFromZero), sbyte.MinValue, sbyte.MaxValue);
                result[o + 4 + j] = unchecked((byte)(sbyte)q);
            }
            vOff += QK_K;
        }
        return result;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float BlockAbsMax(ReadOnlySpan<float> values, int off)
    {
        float amax = 0f;
        for (int j = 0; j < QK_K; j++)
        {
            float a = MathF.Abs(values[off + j]);
            if (a > amax) amax = a;
        }
        return amax;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (float Min, float Max) GroupMinMax(ReadOnlySpan<float> values, int off, int group)
    {
        float gmn = float.MaxValue, gmx = float.MinValue;
        for (int j = 0; j < group; j++)
        {
            float x = values[off + j];
            if (x < gmn) gmn = x;
            if (x > gmx) gmx = x;
        }
        return (gmn, gmx);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Clamp(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);

    /// <summary>Packs 16 six-bit values into 12 bytes (Q3_K scale block).</summary>
    private static void PackSixBitTwelve(ReadOnlySpan<int> sc, byte[] dest, int off)
    {
        for (int j = 0; j < 4; j++)
        {
            dest[off + j] = (byte)((sc[j] & 0x0F) | ((sc[8 + j] & 0x0F) << 4));
            dest[off + 4 + j] = (byte)((sc[4 + j] & 0x0F) | ((sc[12 + j] & 0x0F) << 4));
            dest[off + 8 + j] = (byte)(((sc[j] >> 4) & 0x03)
                                     | (((sc[4 + j] >> 4) & 0x03) << 2)
                                     | (((sc[8 + j] >> 4) & 0x03) << 4)
                                     | (((sc[12 + j] >> 4) & 0x03) << 6));
        }
    }

    /// <summary>Packs 8 scale + 8 min six-bit values into 12 bytes (Q4_K/Q5_K scale block).</summary>
    private static void PackScaleMinK4(ReadOnlySpan<int> sc, ReadOnlySpan<int> mn, byte[] dest, int off)
    {
        for (int j = 0; j < 4; j++)
        {
            dest[off + j] = (byte)((sc[j] & 0x3F) | ((sc[4 + j] >> 4) << 6));
            dest[off + 4 + j] = (byte)((mn[j] & 0x3F) | ((mn[4 + j] >> 4) << 6));
            dest[off + 8 + j] = (byte)((sc[4 + j] & 0x0F) | ((mn[4 + j] & 0x0F) << 4));
        }
    }
}