using System;
using System.IO;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// IQ2_XXS, IQ2_XS and IQ3_XXS decode through codebook grids: a block holds
/// grid indices, one packed scale nibble per 32-value sub-block and four 7-bit
/// sign patterns per sub-block. The three layouts differ only in where those
/// live, so each fixture pins its own offsets against values computed by hand
/// from ggml's <c>dequantize_row_iq2_xxs</c> / <c>_iq2_xs</c> / <c>_iq3_xxs</c>.
/// </summary>
public class IQ23Tests
{
    private const int Qk = 256;

    private static QuantizationOps Ops() => QuantizationFactory.Create(HardwareTier.Scalar);

    private static void SetD(Span<byte> block, float d)
    {
        ushort half = QuantizationKernels.FloatToHalf_F16C(d);
        block[0] = (byte)(half & 0xFF);
        block[1] = (byte)(half >> 8);
    }

    private static float[] Read(byte[] raw, int count)
    {
        var data = new float[count];
        using var reader = new BinaryReader(new MemoryStream(raw, writable: false));
        Ops().ReadFor(raw.Length == 66 ? QuantDType.IQ2_XXS
                   : raw.Length == 74 ? QuantDType.IQ2_XS
                   : raw.Length == 98 ? QuantDType.IQ3_XXS
                   : raw.Length == 82 ? QuantDType.IQ2_S
                   : QuantDType.IQ3_S, reader, data, count);
        return data;
    }

    /// <summary>
    /// Grid entry 0 is eight 8s in IQ2 and four 4s in IQ3, with d = 1 and the
    /// scale nibble at its minimum, so the decoded value is a clean constant and
    /// any offset error shows up as the wrong constant rather than as noise.
    /// </summary>
    [Fact]
    public void IQ2_XXS_DecodesGridScaleAndSigns()
    {
        var raw = new byte[66];
        SetD(raw, 1f);
        // ib 0: grid indices 0 in bytes 2..5, word at 6..9 = 0 -> db = 0.125,
        // sign index 0 -> ksigns[0] = 0b0000000 -> no negatives.
        var data = Read(raw, Qk);
        // Every sub-block indexes grid entry 0 with sign index 0, so all 256
        // values are db * 8 = 1.0.
        for (int j = 0; j < Qk; j++) Assert.Equal(1f, data[j]);

        // Sign index 1 negates values 0 and 7; the scale nibble 0xF multiplies
        // by (0.5 + 15) * 0.25.
        raw[6] = 0x01;
        raw[9] = 0xF0;
        data = Read(raw, Qk);
        float m = (0.5f + 15f) * 0.25f * 8f;
        Assert.Equal(-m, data[0]);
        Assert.Equal(m, data[1]);
        Assert.Equal(-m, data[7]);
        Assert.Equal(m, data[8]); // l = 1 uses sign index 0 again
        Assert.Equal(1f, data[32]); // ib 1 is untouched
    }

    [Fact]
    public void IQ2_XS_DecodesBothScaleNibbles()
    {
        var raw = new byte[74];
        SetD(raw, 1f);
        raw[66] = 0x40; // low nibble 0 -> 0.125, high nibble 4 -> (0.5 + 4) * 0.25

        var data = Read(raw, Qk);

        for (int j = 0; j < 16; j++) Assert.Equal(1f, data[j]);
        for (int j = 16; j < 32; j++) Assert.Equal(9f, data[j]);
        for (int j = 32; j < Qk; j++) Assert.Equal(1f, data[j]); // untouched sub-blocks
    }

    [Fact]
    public void IQ3_XXS_DecodesGridSignsAndScale()
    {
        var raw = new byte[98];
        SetD(raw, 1f);
        // ib 0: qs bytes 2..9 index iq3xxs grid entries 0 (four 4s), the word at
        // 66..69 is 0 -> db = (0.5 + 0) * 0.5, ksigns[0] -> no negatives.
        var data = Read(raw, Qk);
        for (int j = 0; j < Qk; j++) Assert.Equal(1f, data[j]);

        // Sign index 1 negates values 0 and 7 (the low half's first and the high
        // half's last); word nibble 7 sets db = (0.5 + 7) * 0.5.
        raw[66] = 0x01;
        raw[69] = 0x70;
        data = Read(raw, Qk);
        float m = (0.5f + 7f) * 0.5f * 4f;
        Assert.Equal(-m, data[0]);
        Assert.Equal(m, data[1]);
        Assert.Equal(-m, data[7]);
        Assert.Equal(m, data[8]);  // l = 1 shares ib 0's scale and sign index 0
        Assert.Equal(1f, data[32]); // ib 1 is untouched
    }

    [Theory]
    [InlineData(QuantDType.IQ2_XXS, 66)]
    [InlineData(QuantDType.IQ2_XS, 74)]
    [InlineData(QuantDType.IQ3_XXS, 98)]
    [InlineData(QuantDType.IQ2_S, 82)]
    [InlineData(QuantDType.IQ3_S, 110)]
    public void ByteCount_IsTheBlockSizePer256(QuantDType dtype, int blockBytes)
    {
        Assert.Equal(blockBytes, QuantizationOps.GetRawTensorByteCount([Qk, 1], dtype));
        Assert.Equal(2 * blockBytes, QuantizationOps.GetRawTensorByteCount([Qk, 2], dtype));
        Assert.Equal(3 * blockBytes, QuantizationOps.GetRawTensorByteCount([512 + 1, 1], dtype));
        Assert.Equal(Qk, QuantizationOps.GetBlockSize(dtype));
    }

    /// <summary>
    /// The S variants put the ninth (IQ3_S) or ninth and tenth (IQ2_S) index bit
    /// in a <c>qh</c> byte, and IQ3_S uses a different scale bias. Both fixtures
    /// use grid entry 256, whose bytes are pinned GNU db's table values:
    /// IQ2_S -> 25,25,25,25,25,8,25,8 and IQ3_S -> 7,5,9,5.
    /// </summary>
    [Fact]
    public void IQ2_S_DecodesQhIndexBitsAndSigns()
    {
        var raw = new byte[82];
        SetD(raw, 1f);
        raw[66] = 0x01; // qh bit 0 -> index bit 8 for l = 0
        raw[2 + 32] = 0x01; // sign byte 0 -> negate value 0

        var data = Read(raw, Qk);

        float d = 0.125f;
        Assert.Equal(-d * 25, data[0]);
        Assert.Equal(d * 25, data[1]);
        Assert.Equal(d * 8, data[5]);
        Assert.Equal(d * 8, data[7]);
        // Only qh bits 0..1 belong to l = 0, so everything from element 8 on
        // falls back to index 0 (eight 8s).
        for (int j = 8; j < Qk; j++) Assert.Equal(d * 8, data[j]);
    }

    [Fact]
    public void IQ3_S_DecodesScaleSignsAndQhBit()
    {
        var raw = new byte[110];
        SetD(raw, 1f);
        raw[66] = 0x01;  // qh bit 0 -> index bit 8 for the first 4 values
        raw[74] = 0x01;  // first sign byte -> negate value 0
        raw[106] = 0x0F; // low nibble 15 -> db1 = 1 + 2 * 15 for sub-block 0

        var data = Read(raw, Qk);

        float db = 1 + 2 * 15;
        Assert.Equal(-db * 7, data[0]);
        Assert.Equal(db * 5, data[1]);
        Assert.Equal(db * 9, data[2]);
        Assert.Equal(db * 5, data[3]);
        Assert.Equal(db, data[4]);  // grid entry 0 is four 1s
        Assert.Equal(db * 1, data[7]);
        Assert.Equal(1f, data[32]); // sub-block 1 uses the high nibble, 0 -> 1.0
    }

    [Theory]
    [InlineData(QuantDType.IQ2_XXS, 66)]
    [InlineData(QuantDType.IQ2_XS, 74)]
    [InlineData(QuantDType.IQ3_XXS, 98)]
    [InlineData(QuantDType.IQ2_S, 82)]
    [InlineData(QuantDType.IQ3_S, 110)]
    public unsafe void VecDot_MatchesReader(QuantDType dtype, int blockBytes)
    {
        foreach (int inFeatures in new[] { Qk, 300, 97 })
        {
            int nBlocks = (inFeatures + Qk - 1) / Qk;
            var raw = new byte[nBlocks * blockBytes];
            new Random(inFeatures).NextBytes(raw);
            for (int b = 0; b < nBlocks; b++) SetD(raw.AsSpan(b * blockBytes), 0.25f);

            var weights = new float[nBlocks * Qk];
            using (var reader = new BinaryReader(new MemoryStream(raw, writable: false)))
                Ops().ReadFor(dtype, reader, weights, weights.Length);

            var input = new float[inFeatures];
            var rng = new Random(13);
            for (int i = 0; i < inFeatures; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

            double expected = 0;
            for (int i = 0; i < inFeatures; i++) expected += input[i] * weights[i];

            float actual;
            fixed (float* pIn = input) fixed (byte* pW = raw)
                actual = Ops().VecDotFor(dtype, pIn, pW, 0, inFeatures);

            Assert.True(Math.Abs(actual - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                $"{dtype} {inFeatures}: expected {expected}, got {actual}");
        }
    }

    /// <summary>
    /// End to end for a real layer: the compound must resolve, the 200-or-so
    /// byte shape guard must accept the buffer, and the forward must agree with
    /// the reader's reconstruction.
    /// </summary>
    [Theory]
    [InlineData(QuantDType.IQ2_XXS, 66)]
    [InlineData(QuantDType.IQ2_XS, 74)]
    [InlineData(QuantDType.IQ3_XXS, 98)]
    [InlineData(QuantDType.IQ2_S, 82)]
    [InlineData(QuantDType.IQ3_S, 110)]
    public void Forward_MatchesDequantizedReference(QuantDType dtype, int blockBytes)
    {
        const int k = Qk;
        const int n = 8;
        var raw = new byte[blockBytes * n];
        new Random(77).NextBytes(raw);
        for (int row = 0; row < n; row++) SetD(raw.AsSpan(row * blockBytes), 0.5f);

        var weights = new float[k * n];
        using (var reader = new BinaryReader(new MemoryStream(raw, writable: false)))
            Ops().ReadFor(dtype, reader, weights, weights.Length);

        var input = new float[k];
        var rng = new Random(3);
        for (int i = 0; i < k; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: false);
        using var layer = LinearLayerFactory.Create(
            "grid_proj", k, n, bias: false, weight: null, biasTensor: null, dtype, mapping);
        layer.SetRawWeight(raw);

        using var x = new Tensor<float>(1, k);
        input.CopyTo(x.Data);
        using var y = layer.Forward(x);

        for (int col = 0; col < n; col++)
        {
            double expected = 0;
            for (int i = 0; i < k; i++) expected += input[i] * weights[col * k + i];
            Assert.True(Math.Abs(y.Data[col] - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                $"{dtype} col {col}: expected {expected}, got {y.Data[col]}");
        }
    }
}
