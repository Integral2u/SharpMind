// vecdot_ref.cs — Standalone C# reference for SharpMind VecDot validation
// MIT License — Algorithms derived from ggml-common.h and ggml-cpu/quants.c (ggml-org/llama.cpp, MIT)
// Built from scratch with independent implementation to catch block-layout and arithmetic bugs.

using System.Numerics;
using System.Runtime.InteropServices;

namespace VecDotRef;

class Program
{
    // Constants matching SharpMind
    const int QK = 32;
    const int QK_K = 256;

    // Enum matching SharpMind.QuantDType (exact values)
    enum QuantDType
    {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
        Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15
    }

    // HalfToFloat matching SharpMind HalfToFloat_Scalar
    static unsafe float HalfToFloat(ushort h)
    {
        int exp5 = (h >> 10) & 0x1F;
        if (exp5 == 0)
        {
            uint mant10 = (uint)(h & 0x3FF);
            if (mant10 == 0)
                return (h & 0x8000) == 0 ? 0f : -0f;
            int lz = BitOperations.LeadingZeroCount(mant10);
            int k = 31 - lz;
            uint e = (uint)(k + 103);
            uint m = (mant10 - (1u << k)) << (23 - k);
            uint bits = ((uint)(h & 0x8000) << 16) | (e << 23) | m;
            return *(float*)&bits;
        }
        if (exp5 == 31)
        {
            if ((h & 0x3FF) == 0)
                return (h & 0x8000) != 0 ? float.NegativeInfinity : float.PositiveInfinity;
            return float.NaN;
        }
        uint eBits = (uint)(exp5 + 112);
        uint mMant = (uint)(h & 0x3FF) << 13;
        uint bitsNrm = ((uint)(h & 0x8000) << 16) | (eBits << 23) | mMant;
        return *(float*)&bitsNrm;
    }

    // GetScaleMinK4 helpers
    static int GetScale(int j, byte[] scales)
    {
        if (j < 4) return scales[j] & 0x3F;
        return (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
    }
    static int GetMin(int j, byte[] scales)
    {
        if (j < 4) return scales[j + 4] & 0x3F;
        return (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
    }

    // ===== VecDot functions =====

    static float VecDotQ4_0(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 18;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            var qs = rawWeights.Slice(blockOff + 2, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * ((q - 8) * d);
            }
        }
        return (float)sum;
    }

    static float VecDotQ4_1(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 20;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float m = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            var qs = rawWeights.Slice(blockOff + 4, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                int q = (i < QK / 2) ? (qs[i] & 0x0F) : (qs[i - QK / 2] >> 4);
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    static float VecDotQ5_0(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 22;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            uint qh = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 2, 4));
            var qs = rawWeights.Slice(blockOff + 6, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            int half = QK / 2;
            for (int i = 0; i < blockEnd; i++)
            {
                int h4 = ((int)(qh >> i) & 1) << 4;
                int nib = (i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4);
                int q = nib | h4;
                sum += input[b * QK + i] * ((q - 16) * d);
            }
        }
        return (float)sum;
    }

    static float VecDotQ5_1(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 24;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float m = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            uint qh = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 4, 4));
            var qs = rawWeights.Slice(blockOff + 8, 16);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            int half = QK / 2;
            for (int i = 0; i < blockEnd; i++)
            {
                int xh = (int)((qh >> i) & 1) << 4;
                int q = ((i < half) ? (qs[i] & 0x0F) : (qs[i - half] >> 4)) | xh;
                sum += input[b * QK + i] * (q * d + m);
            }
        }
        return (float)sum;
    }

    static float VecDotQ8_0(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 2 + i];
                sum += input[b * QK + i] * (val * d);
            }
        }
        return (float)sum;
    }

    static float VecDotQ8_1(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 36;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = col * nBlocks * blockBytes + b * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            for (int i = 0; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 4 + i];
                sum += input[b * QK + i] * (val * d);
            }
        }
        return (float)sum;
    }

    static float VecDotQ2_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 84;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float dSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 80, 2)));
            float minSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 82, 2)));
            byte[] scales = rawWeights.Slice(blockOff, 16).ToArray();
            var qs = rawWeights.Slice(blockOff + 16, 64);

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 8 + j * 2;
                    int s0 = scales[isc] & 0x0F;
                    int m0 = scales[isc] >> 4;
                    for (int l = 0; l < 16 && basePos + l < blockEnd; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += input[b * QK_K + idx - colBlockStart] * (s0 * v * dSuper - m0 * minSuper);
                    }
                    int s1 = scales[isc + 1] & 0x0F;
                    int m1 = scales[isc + 1] >> 4;
                    for (int l = 0; l < 16 && basePos + 16 + l < blockEnd; l++)
                    {
                        int idx = basePos + 16 + l;
                        int qsByte = (idx / 128) * 32 + (idx % 32);
                        int qsShift = ((idx % 128) / 32) * 2;
                        int v = (qs[qsByte] >> qsShift) & 3;
                        sum += input[b * QK_K + idx - colBlockStart] * (s1 * v * dSuper - m1 * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    static float VecDotQ3_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 110;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float dAll = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 108, 2)));

            // Unpack scales
            uint kmask1 = 0x03030303u;
            uint kmask2 = 0x0f0f0f0fu;
            Span<byte> scaleBuf = stackalloc byte[16];
            uint aux0 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 96, 4));
            uint aux1 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 100, 4));
            uint aux2 = BitConverter.ToUInt32(rawWeights.Slice(blockOff + 104, 4));
            uint tmp = aux2;
            BitConverter.TryWriteBytes(scaleBuf.Slice(0, 4), (aux0 & kmask2) | (((tmp >> 0) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(4, 4), (aux1 & kmask2) | (((tmp >> 2) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(8, 4), ((aux0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4));
            BitConverter.TryWriteBytes(scaleBuf.Slice(12, 4), ((aux1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4));
            var sc8 = MemoryMarshal.Cast<byte, sbyte>(scaleBuf);

            var hmask = rawWeights.Slice(blockOff, 32);
            var qs = rawWeights.Slice(blockOff + 32, 64);

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
            {
                int qsByte = (i / 128) * 32 + (i % 32);
                int qsShift = ((i % 128) / 32) * 2;
                int s2 = (qs[qsByte] >> qsShift) & 3;
                int hBit = (hmask[i % 32] >> (i / 32)) & 1;
                int actual = s2 - (hBit == 0 ? 4 : 0);
                float val = dAll * (sc8[i / 16] - 32) * actual;
                sum += input[b * QK_K + i - colBlockStart] * val;
            }
        }
        return (float)sum;
    }

    static float VecDotQ4_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 144;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float dSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float minSuper = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            byte[] scales = rawWeights.Slice(blockOff + 4, 12).ToArray();
            var qs = rawWeights.Slice(blockOff + 16, 128);

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
            {
                for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                {
                    int basePos = n16 + j * 32;
                    int isc = (n16 / 128) * 4 + j;
                    float s = GetScale(isc, scales);
                    float m = GetMin(isc, scales);
                    for (int l = 0; l < 32 && basePos + l < blockEnd; l++)
                    {
                        int idx = basePos + l;
                        int qsByte = (idx / 64) * 32 + (idx % 32);
                        int qsShift = ((idx % 64) / 32) * 4;
                        int v = (qs[qsByte] >> qsShift) & 0x0F;
                        sum += input[b * QK_K + idx - colBlockStart] * (s * v * dSuper - m * minSuper);
                    }
                }
            }
        }
        return (float)sum;
    }

    static float VecDotQ5_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 176;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff, 2)));
            float min = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 2, 2)));
            byte[] scales = rawWeights.Slice(blockOff + 4, 12).ToArray();
            var qh = rawWeights.Slice(blockOff + 16, 32);
            var qs = rawWeights.Slice(blockOff + 48, 128);

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
            {
                int sc = GetScale(i / 32, scales);
                int mn = GetMin(i / 32, scales);
                int idx32 = i % 32;
                int group64 = i / 64;
                int half = (i % 64) / 32;
                int bitPos = group64 * 2 + half;
                int hAdd = ((qh[idx32] & (1 << bitPos)) != 0) ? 16 : 0;
                int q5 = (half == 0) ? (qs[group64 * 32 + idx32] & 0x0F) : (qs[group64 * 32 + idx32] >> 4);
                q5 |= hAdd;
                sum += input[b * QK_K + i - colBlockStart] * (sc * q5 * d - mn * min);
            }
        }
        return (float)sum;
    }

    static float VecDotQ6_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 210;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float d = HalfToFloat(BitConverter.ToUInt16(rawWeights.Slice(blockOff + 208, 2)));
            var ql = rawWeights.Slice(blockOff, 128);
            var qh = rawWeights.Slice(blockOff + 128, 64);
            var scales = MemoryMarshal.Cast<byte, sbyte>(rawWeights.Slice(blockOff + 192, 16));

            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int nOff = curBlockStart; nOff < blockEnd; nOff += 128)
            {
                var pql = ql.Slice(nOff == 0 ? 0 : 64, 64);
                var pqh = qh.Slice(nOff == 0 ? 0 : 32, 32);
                var psc = scales.Slice(nOff == 0 ? 0 : 8, 8);

                int halfRem = Math.Min(128, blockEnd - nOff);
                for (int l = 0; l < 32 && l < halfRem; l++)
                {
                    int is_ = l / 16;
                    int q1v = (pql[l] & 0x0F) | ((pqh[l] & 0x03) << 4);
                    int q2v = (pql[l + 32] & 0x0F) | (((pqh[l] >> 2) & 0x03) << 4);
                    int q3v = ((pql[l] >> 4) & 0x0F) | (((pqh[l] >> 4) & 0x03) << 4);
                    int q4v = ((pql[l + 32] >> 4) & 0x0F) | (((pqh[l] >> 6) & 0x03) << 4);

                    int i1 = b * QK_K + nOff + l - colBlockStart;
                    int i2 = b * QK_K + nOff + l + 32 - colBlockStart;

                    if (i2 >= b * QK_K + blockEnd - colBlockStart)
                    {
                        if (i1 < b * QK_K + blockEnd - colBlockStart)
                            sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                        break;
                    }

                    int i3 = b * QK_K + nOff + l + 64 - colBlockStart;
                    int i4 = b * QK_K + nOff + l + 96 - colBlockStart;

                    sum += input[i1] * (d * psc[is_ + 0] * (q1v - 32));
                    sum += input[i2] * (d * psc[is_ + 2] * (q2v - 32));
                    sum += input[i3] * (d * psc[is_ + 4] * (q3v - 32));
                    sum += input[i4] * (d * psc[is_ + 6] * (q4v - 32));
                }
            }
        }
        return (float)sum;
    }

    static float VecDotQ8_K(ReadOnlySpan<float> input, ReadOnlySpan<byte> rawWeights, int col, int inFeatures)
    {
        const int blockBytes = 292;
        int startBlock = (col * inFeatures) / QK_K;
        int colBlockStart = col * inFeatures % QK_K;
        int nBlocks = (inFeatures + QK_K - 1) / QK_K;
        double sum = 0;
        for (int b = 0; b < nBlocks; b++)
        {
            int blockOff = (startBlock + b) * blockBytes;
            float d = BitConverter.ToSingle(rawWeights.Slice(blockOff, 4));
            int curBlockStart = (b == 0) ? colBlockStart : 0;
            int blockEnd = Math.Min(QK_K, inFeatures + colBlockStart - b * QK_K);
            for (int i = curBlockStart; i < blockEnd; i++)
            {
                sbyte val = (sbyte)rawWeights[blockOff + 4 + i];
                sum += input[b * QK_K + i - colBlockStart] * (val * d);
            }
        }
        return (float)sum;
    }

    static void WriteFloat(BinaryWriter w, float f) => w.Write(f);
    static void WriteInt(BinaryWriter w, int i) => w.Write(i);

    static int Main(string[] args)
    {
        Stream inStream;
        if (args.Length > 0)
            inStream = File.OpenRead(args[0]);
        else
            inStream = Console.OpenStandardInput();
        using var reader = new BinaryReader(inStream);

        // Read header
        int dtype = reader.ReadInt32();
        int inFeatures = reader.ReadInt32();
        int col = reader.ReadInt32();

        // Read input floats
        float[] input = new float[inFeatures];
        for (int i = 0; i < inFeatures; i++)
            input[i] = reader.ReadSingle();

        // Determine block size and read weights
        int blockBytes = dtype switch
        {
            (int)QuantDType.Q4_0 => 18,
            (int)QuantDType.Q4_1 => 20,
            (int)QuantDType.Q5_0 => 22,
            (int)QuantDType.Q5_1 => 24,
            (int)QuantDType.Q8_0 => 34,
            (int)QuantDType.Q8_1 => 36,
            (int)QuantDType.Q2_K => 84,
            (int)QuantDType.Q3_K => 110,
            (int)QuantDType.Q4_K => 144,
            (int)QuantDType.Q5_K => 176,
            (int)QuantDType.Q6_K => 210,
            (int)QuantDType.Q8_K => 292,
            _ => throw new InvalidOperationException($"Unknown dtype: {dtype}")
        };
        int qk = dtype >= (int)QuantDType.Q2_K ? QK_K : QK;
        int nBlocks = (inFeatures + qk - 1) / qk;
        var remaining = new MemoryStream();
        reader.BaseStream.CopyTo(remaining);
        byte[] weights = remaining.ToArray();

        // Compute
        float result = (QuantDType)dtype switch
        {
            QuantDType.Q4_0 => VecDotQ4_0(input, weights, col, inFeatures),
            QuantDType.Q4_1 => VecDotQ4_1(input, weights, col, inFeatures),
            QuantDType.Q5_0 => VecDotQ5_0(input, weights, col, inFeatures),
            QuantDType.Q5_1 => VecDotQ5_1(input, weights, col, inFeatures),
            QuantDType.Q8_0 => VecDotQ8_0(input, weights, col, inFeatures),
            QuantDType.Q8_1 => VecDotQ8_1(input, weights, col, inFeatures),
            QuantDType.Q2_K => VecDotQ2_K(input, weights, col, inFeatures),
            QuantDType.Q3_K => VecDotQ3_K(input, weights, col, inFeatures),
            QuantDType.Q4_K => VecDotQ4_K(input, weights, col, inFeatures),
            QuantDType.Q5_K => VecDotQ5_K(input, weights, col, inFeatures),
            QuantDType.Q6_K => VecDotQ6_K(input, weights, col, inFeatures),
            QuantDType.Q8_K => VecDotQ8_K(input, weights, col, inFeatures),
            _ => throw new InvalidOperationException()
        };

        Console.WriteLine("{0:F9}", result);
        return 0;
    }
}
