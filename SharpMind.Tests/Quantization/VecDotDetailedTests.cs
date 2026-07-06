using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Tests.Quantization;

public class VecDotDetailedTests
{
    public static IEnumerable<object[]> AllTiers()
    {
        yield return new object[] { HardwareTier.Scalar };
        if (Sse.IsSupported)
        {
            yield return new object[] { HardwareTier.SSE };
            if (Avx2.IsSupported)
            {
                yield return new object[] { HardwareTier.AVX2 };
                if (Fma.IsSupported)
                    yield return new object[] { HardwareTier.FMA };
            }
        }
    }

    /// <summary>
    /// Creates a realistic Q4_K block with non-trivial values, dequantizes via ReadQ4K,
    /// computes expected dot product with a random input, and compares with VecDotQ4K.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllTiers))]
    public unsafe void VecDotQ4K_MatchesDequant(HardwareTier tier)
    {
        // Build a Q4_K block with known values:
        // - dSuper = 2.0
        // - minSuper = 1.0
        // - Sub-block 0: scale=3, min=1
        // - All qs values in sub-block 0 = 5
        // - All other sub-blocks: scale=0, min=0, qs=0
        byte[] block = new byte[144];

        // dSuper = 2.0 (half: 0x4000)
        block[0] = 0x00; block[1] = 0x40;

        // minSuper = 1.0 (half: 0x3C00)
        block[2] = 0x00; block[3] = 0x3C;

        // GetScaleMinK4: scales[j] & 0x3F = scale, scales[j+4] & 0x3F = min
        // scale0=3 at scales[0], min0=1 at scales[4]
        block[4] = 0x03;
        block[8] = 0x01;

        // qs[0..31] for first 64 elements: each nibble = 5
        for (int i = 0; i < 32; i++)
            block[16 + i] = 0x55;

        // Expected dequantized value for first sub-block (elements 0..31):
        // scale * q * dSuper - min * minSuper = 3 * 5 * 2.0 - 1 * 1.0 = 30 - 1 = 29

        // Dequantize manually
        float[] dequant = new float[256];
        fixed (byte* pBlock = block)
        {
            using var ms = new MemoryStream(block);
            using var reader = new BinaryReader(ms);
            GgufLoaderFactory.Default.ReadQ4K(reader, dequant, 256);
        }

        // Compute expected dot product with all-ones input
        float[] input = new float[256];
        for (int i = 0; i < 256; i++) input[i] = 1.0f;

        float expected = 0;
        for (int i = 0; i < 256; i++) expected += input[i] * dequant[i];

        // First 32 elements: 29 each = 928
        // Remaining elements: all zero
        Assert.Equal(928.0f, expected, 4);

        // Now run VecDotQ4K on all tiers and compare
        var qOps = QuantizationFactory.Create(tier);
        float result;
        fixed (float* pIn = input)
        fixed (byte* pBlock = block)
        {
            result = qOps.VecDotQ4K(pIn, pBlock, 0, 256);
        }

        Assert.Equal(expected, result, 2);
    }

    /// <summary>
    /// Tests Q2_K VecDot with non-trivial dSuper and minSuper values.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllTiers))]
    public unsafe void VecDotQ2K_NonTrivial(HardwareTier tier)
    {
        byte[] block = new byte[84];

        // dSuper = 2.0 (half: 0x4000 at offset 80)
        block[80] = 0x00; block[81] = 0x40;

        // minSuper = 1.0 (half: 0x3C00 at offset 82)
        block[82] = 0x00; block[83] = 0x3C;

        // scales[0] = (min0 << 4) | scale0, where each is 4-bit
        // scale0=2, min0=1 → scales[0] = 0x12
        block[0] = 0x12;

        // qs[0..15] for first 16 elements: each byte has 2-bit value 3 (0xFF & 3 = 3)
        // GGML Q2_K layout: qs[0..15] at shift 0 hold elements 0..15
        for (int i = 0; i < 16; i++) block[16 + i] = 0xFF;

        // Dequantize manually
        float[] dequant = new float[256];
        fixed (byte* pBlock = block)
        {
            using var ms = new MemoryStream(block);
            using var reader = new BinaryReader(ms);
            GgufLoaderFactory.Default.ReadQ2K(reader, dequant, 256);
        }

        float[] input = new float[256];
        for (int i = 0; i < 256; i++) input[i] = 1.0f;

        float expected = 0;
        for (int i = 0; i < 256; i++) expected += input[i] * dequant[i];

        // Element 0: scale0 * q * dSuper - min0 * minSuper = 2 * 3 * 2 - 1 * 1 = 12 - 1 = 11
        // Elements 0..15 all have q=3 → 11 each = 176
        // Elements 16+: q=0, scale=0, min=0 → 0
        Assert.Equal(176.0f, expected, 4);

        var qOps = QuantizationFactory.Create(tier);
        float result;
        fixed (float* pIn = input)
        fixed (byte* pBlock = block)
        {
            result = qOps.VecDotQ2K(pIn, pBlock, 0, 256);
        }

        Assert.Equal(expected, result, 2);
    }

    /// <summary>
    /// Tests Q5_K VecDot with non-trivial values.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllTiers))]
    public unsafe void VecDotQ5K_NonTrivial(HardwareTier tier)
    {
        byte[] block = new byte[176];

        // d = 2.0 (half: 0x4000 at offset 0)
        block[0] = 0x00; block[1] = 0x40;

        // min = 1.0 (half: 0x3C00 at offset 2)
        block[2] = 0x00; block[3] = 0x3C;

        // GetScaleMinK4: scales[j] & 0x3F = scale, scales[j+4] & 0x3F = min
        // scale=4 at scales[0], min=2 at scales[4]
        block[4] = 0x04;
        block[8] = 0x02;

        // qh = 0 (no high bits)
        // Q5_K GGML layout: qs[g*32 + l] for group g, index l (0..31)
        //   Low nibble = elements g*64 + l, high nibble = elements g*64 + l + 32
        // qs[0] = 0x01 → element 0 low nibble=1
        // qs[1] = 0x05 → element 1 low nibble=5
        block[48] = 0x01;
        block[49] = 0x05;

        float[] dequant = new float[256];
        fixed (byte* pBlock = block)
        {
            using var ms = new MemoryStream(block);
            using var reader = new BinaryReader(ms);
            GgufLoaderFactory.Default.ReadQ5_K(reader, dequant, 256);
        }

        Assert.Equal(6.0f, dequant[0], 4);
        Assert.Equal(38.0f, dequant[1], 4);

        float[] input = new float[256];
        for (int i = 0; i < 256; i++) input[i] = 1.0f;

        float expected = dequant.Sum();

        var qOps = QuantizationFactory.Create(tier);
        float result;
        fixed (float* pIn = input)
        fixed (byte* pBlock = block)
        {
            result = qOps.VecDotQ5K(pIn, pBlock, 0, 256);
        }

        Assert.Equal(expected, result, 1);
    }

    /// <summary>
    /// Tests Q6_K VecDot with non-trivial values.
    /// Uses same pattern as Q6KRefTests.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllTiers))]
    public unsafe void VecDotQ6K_NonTrivial(HardwareTier tier)
    {
        byte[] block = new byte[210];

        // d = 1.0 (half: 0x3C00 at offset 208) — same as reference test
        block[208] = 0x00; block[209] = 0x3C;

        // sc[0] = 32, rest = 0
        block[192] = 32;

        // ql[0] = 0x01 (low nibble = 1)
        block[0] = 0x01;

        float[] dequant = new float[256];
        using var ms = new MemoryStream(block);
        using var reader = new BinaryReader(ms);
        GgufLoaderFactory.Default.ReadQ6K(reader, dequant, 256);

        // Element 0: d * sc[0] * (1 - 32) = 1.0 * 32 * (-31) = -992
        Assert.Equal(-992.0f, dequant[0], 2);

        float[] input = new float[256];
        for (int i = 0; i < 256; i++) input[i] = 1.0f;

        float expected = dequant.Sum();

        var qOps = QuantizationFactory.Create(tier);
        float result;
        fixed (float* pIn = input)
        fixed (byte* pBlock = block)
        {
            result = qOps.VecDotQ6K(pIn, pBlock, 0, 256);
        }

        Assert.Equal(expected, result, 2);
    }

    /// <summary>
    /// Tests Q3_K VecDot with non-trivial values.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllTiers))]
    public unsafe void VecDotQ3K_NonTrivial(HardwareTier tier)
    {
        byte[] block = new byte[110];

        // d = 2.0 (half: 0x4000 at offset 0)
        block[0] = 0x00; block[1] = 0x40;

        // qs all zero → actual = 0 - 4 = -4 (since hmask=0)
        // hmask all zero (already zero-initialized)
        // scales all zero (already zero-initialized)

        float[] dequant = new float[256];
        using var ms = new MemoryStream(block);
        using var reader = new BinaryReader(ms);
        GgufLoaderFactory.Default.ReadQ3_K(reader, dequant, 256);

        float[] input = new float[256];
        for (int i = 0; i < 256; i++) input[i] = 1.0f;

        float expected = 0;
        for (int i = 0; i < 256; i++) expected += input[i] * dequant[i];

        var qOps = QuantizationFactory.Create(tier);
        float result;
        fixed (float* pIn = input)
        fixed (byte* pBlock = block)
        {
            result = qOps.VecDotQ3K(pIn, pBlock, 0, 256);
        }

        Assert.Equal(expected, result, 2);
    }

    /// <summary>
    /// Tests Q5_1 VecDot with non-trivial values.
    /// </summary>
    [Theory]
    [MemberData(nameof(AllTiers))]
    public unsafe void VecDotQ5_1_NonTrivial(HardwareTier tier)
    {
        byte[] block = new byte[24];

        // d = 2.0 (half: 0x4000 at offset 0)
        block[0] = 0x00; block[1] = 0x40;

        // min = 1.0 (half: 0x3C00 at offset 2)
        block[2] = 0x00; block[3] = 0x3C;

        // qh = 0 (no high bits for first 8 elements)
        // qs[0] = 0x53 → low nibble=3 (element 0), high nibble=5 (element 1)
        block[8] = 0x53;

        // Expected: element 0 = 3 * 2 + 1 = 7, element 1 = 5 * 2 + 1 = 11

        float[] dequant = new float[32];
        using var ms = new MemoryStream(block);
        using var reader = new BinaryReader(ms);
        GgufLoaderFactory.Default.ReadQ5_1(reader, dequant, 32);

        Assert.Equal(7.0f, dequant[0], 4);
        Assert.Equal(11.0f, dequant[1], 4);

        float[] input = new float[32];
        for (int i = 0; i < 32; i++) input[i] = 1.0f;

        float expected = dequant.Sum();

        var qOps = QuantizationFactory.Create(tier);
        float result;
        fixed (float* pIn = input)
        fixed (byte* pBlock = block)
        {
            result = qOps.VecDotQ5_1(pIn, pBlock, 0, 32);
        }

        Assert.Equal(expected, result, 1);
    }

    /// <summary>
    /// End-to-end test: loads a real Q2_K weight from a GGUF file
    /// and verifies VecDot matches the dequant+dot product.
    /// </summary>
    [Fact]
    public unsafe void VecDotQ2K_MatchesRealGgufData()
    {
        string ggufPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0.5b-instruct-q2_k.gguf";
        if (!File.Exists(ggufPath)) return;

        var meta = GgufLoaderFactory.Default.LoadMeta(ggufPath);
        var tensor = meta.Tensors.First(t => t.Name == "blk.0.attn_q.weight");

        int[] shape = tensor.Shape;
        int inF = shape[0], outF = shape[1];
        int rawSize = (int)GgufLoaderFactory.Default.GetRawTensorByteCount(shape, tensor.Dtype);

        byte[] rawData = new byte[rawSize];

        using (var fs = File.OpenRead(ggufPath))
        {
            fs.Position = meta.DataOffset + tensor.Offset;
            fs.ReadExactly(rawData);
        }

        var rng = new Random(42);
        float[] input = new float[inF];
        for (int i = 0; i < inF; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        const int QK_K = 256;

        var fmt = tensor.Dtype;
        bool isQ2K = fmt is GgufDtype.Q2_K or GgufDtype.Q2_K_S;
        int blkBytes = isQ2K ? 84 : 144;
        int nBlksPerCol = (inF + QK_K - 1) / QK_K;

        foreach (int col in new[] { 0, 1, outF / 2, outF - 1 })
        {
            if (col >= outF) continue;

            float expected = 0;
            int colBlockStart = (col * inF) % QK_K;
            int colStartBlock = (col * inF) / QK_K;
            for (int b = 0; b < nBlksPerCol; b++)
            {
                int curBlockStart = (b == 0) ? colBlockStart : 0;
                int blockByteOff = (colStartBlock + b) * blkBytes;
                int blockEnd = Math.Min(QK_K, inF + colBlockStart - b * QK_K);
                fixed (byte* pBlock = &rawData[blockByteOff])
                {
                    if (isQ2K)
                    {
                        float dSuper = GgufLoaderFactory.Default.HalfToFloat(Unsafe.ReadUnaligned<ushort>(pBlock + 80));
                        float minSuper = GgufLoaderFactory.Default.HalfToFloat(Unsafe.ReadUnaligned<ushort>(pBlock + 82));
                        byte* scales = pBlock;
                        byte* qs = pBlock + 16;
                        for (int n16 = curBlockStart; n16 < blockEnd; n16 += 128)
                        {
                            for (int j = 0; j < 4 && n16 + j * 32 < blockEnd; j++)
                            {
                                int basePos = n16 + j * 32;
                                int isc = (n16 / 128) * 8 + j * 2;
                                float s0 = scales[isc] & 0x0F;
                                float m0 = scales[isc] >> 4;
                                for (int l = 0; l < 16 && basePos + l < blockEnd; l++)
                                {
                                    int idx = basePos + l;
                                    int qsByte = (idx / 128) * 32 + (idx % 32);
                                    int qsShift = ((idx % 128) / 32) * 2;
                                    int v = (qs[qsByte] >> qsShift) & 3;
                                    expected += input[(b * QK_K + idx) - colBlockStart] * (s0 * v * dSuper - m0 * minSuper);
                                }
                                float s1 = scales[isc + 1] & 0x0F;
                                float m1 = scales[isc + 1] >> 4;
                                for (int l = 0; l < 16 && basePos + 16 + l < blockEnd; l++)
                                {
                                    int idx = basePos + 16 + l;
                                    int qsByte = (idx / 128) * 32 + (idx % 32);
                                    int qsShift = ((idx % 128) / 32) * 2;
                                    int v = (qs[qsByte] >> qsShift) & 3;
                                    expected += input[(b * QK_K + idx) - colBlockStart] * (s1 * v * dSuper - m1 * minSuper);
                                }
                            }
                        }
                    }
                    else
                    {
                        float dSuper = GgufLoaderFactory.Default.HalfToFloat(Unsafe.ReadUnaligned<ushort>(pBlock));
                        float minSuper = GgufLoaderFactory.Default.HalfToFloat(Unsafe.ReadUnaligned<ushort>(pBlock + 2));
                        byte* scaleSpan = pBlock + 4;
                        byte* qs = pBlock + 16;
                        for (int j = curBlockStart; j < blockEnd; j += 64)
                        {
                            int idx = j / 64;
                            int scOff = idx * 2;
                            GetScaleMinK4(scOff, scaleSpan, out byte sc0, out byte m0v);
                            GetScaleMinK4(scOff + 1, scaleSpan, out byte sc1, out byte m1v);
                            int qOff = (j / 64) * 32;
                            int lim1 = Math.Min(32, blockEnd - j);
                            for (int l = 0; l < lim1; l++)
                            {
                                float val = (sc0 * (qs[qOff + l] & 0x0F) * dSuper) - (m0v * minSuper);
                                expected += input[(b * QK_K + j + l) - colBlockStart] * val;
                            }
                            int lim2 = Math.Min(32, blockEnd - j - 32);
                            for (int l = 0; l < lim2; l++)
                            {
                                float val = (sc1 * (qs[qOff + l] >> 4) * dSuper) - (m1v * minSuper);
                                expected += input[(b * QK_K + j + 32 + l) - colBlockStart] * val;
                            }
                        }
                    }
                }
            }

            float result;
            fixed (float* pIn = input)
            fixed (byte* pRaw = rawData)
            {
                if (isQ2K)
                    result = qOps.VecDotQ2K(pIn, pRaw, col, inF);
                else
                    result = qOps.VecDotQ4K(pIn, pRaw, col, inF);
            }

            double relDiff = Math.Abs((double)result - expected) / Math.Max(1.0, Math.Abs(expected));
            Assert.True(relDiff < 0.0001, $"col={col}: expected={expected}, VecDot={result}, relDiff={relDiff}");
        }
    }

    private static unsafe void GetScaleMinK4(int idx, byte* scales, out byte d, out byte m)
    {
        if (idx < 4)
        {
            d = (byte)(scales[idx] & 0x3F);
            m = (byte)(scales[idx + 4] & 0x3F);
        }
        else
        {
            d = (byte)((scales[idx + 4] & 0x0F) | ((scales[idx - 4] >> 6) << 4));
            m = (byte)((scales[idx + 4] >> 4) | ((scales[idx] >> 6) << 4));
        }
    }
}
