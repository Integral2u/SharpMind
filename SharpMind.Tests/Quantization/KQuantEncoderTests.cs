using SharpMind.Core;
using SharpMind.Core.Quantization;
using System.IO;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// Round-trip validation for the K-quant encoders in
/// <see cref="TensorQuantizer"/>. Each test quantizes a deterministic float
/// buffer and reads it back with the corresponding SharpMind decoder,
/// asserting both the recovered weights (dot product vs the F32 reference)
/// and the raw byte layouts the encoders emit.
/// </summary>
public class KQuantEncoderTests
{
    public static IEnumerable<object[]> QuantLevels()
    {
        yield return new object[] { QuantDType.Q2_K };
        yield return new object[] { QuantDType.Q3_K };
        yield return new object[] { QuantDType.Q4_K };
        yield return new object[] { QuantDType.Q5_K };
        yield return new object[] { QuantDType.Q6_K };
        yield return new object[] { QuantDType.Q8_K };
    }

    private static int BlockBytesFor(QuantDType dtype) => dtype switch
    {
        QuantDType.Q2_K => 84,
        QuantDType.Q3_K => 110,
        QuantDType.Q4_K => 144,
        QuantDType.Q5_K => 176,
        QuantDType.Q6_K => 210,
        QuantDType.Q8_K => 292,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
    };

    private static float MaxRelError(QuantDType dtype) => dtype switch
    {
        QuantDType.Q8_K => 0.010f,
        QuantDType.Q6_K => 0.040f,
        QuantDType.Q5_K => 0.080f,
        QuantDType.Q4_K => 0.120f,
        QuantDType.Q3_K => 0.250f,
        QuantDType.Q2_K => 0.350f,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype)),
    };

    [Theory]
    [MemberData(nameof(QuantLevels))]
    public void RoundTrip_WeightFidelityWithinTolerance(QuantDType dtype)
    {
        const int qk = 256;
        const int nBlocks = 4;
        const int nCols = 3;
        int inF = nBlocks * qk;

        var rng = new Random(1234);
        var weights = new float[nCols * inF];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)rng.NextDouble() * 2f - 1f;

        var raw = TensorQuantizer.Quantize(weights, [nCols * nBlocks, qk], dtype);

        for (int c = 0; c < nCols; c++)
        {
            float[] roundTripped = ReadColumn(raw, dtype, c, inF);

            double errSq = 0, sigSq = 0;
            for (int i = 0; i < inF; i++)
            {
                float e = weights[c * inF + i] - roundTripped[i];
                errSq += e * e;
                float s = weights[c * inF + i];
                sigSq += s * s;
            }
            double rel = Math.Sqrt(errSq / Math.Max(1e-30, sigSq));
            Assert.True(rel < MaxRelError(dtype),
                $"{dtype} col {c}: relative weight RMS error {rel:P3} exceeds tolerance");
        }
    }

    [Theory]
    [MemberData(nameof(QuantLevels))]
    public void ZeroBlock_EncodesToZeroLayout(QuantDType dtype)
    {
        var values = new float[256];
        var raw = TensorQuantizer.Quantize(values, [256], dtype);
        Assert.Equal(BlockBytesFor(dtype), raw.Length);

        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        var read = new float[256];
        using var ms = new MemoryStream(raw);
        using var reader = new BinaryReader(ms);
        qOps.ReadFor(dtype, reader, read, 256);
        Assert.All(read, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void KQuant_RejectsLengthNotMultipleOf256()
    {
        var values = Enumerable.Range(0, 300).Select(i => (float)i).ToArray();
        foreach (var dtype in new[] { QuantDType.Q2_K, QuantDType.Q4_K, QuantDType.Q8_K })
            Assert.Throws<InvalidOperationException>(() => TensorQuantizer.Quantize(values, [300], dtype));
    }

    private static float[] ReadColumn(byte[] raw, QuantDType dtype, int col, int n)
    {
        int blockBytes = BlockBytesFor(dtype);
        var slice = new byte[blockBytes * (n / 256)];
        Array.Copy(raw, col * slice.Length, slice, 0, slice.Length);

        using var ms = new MemoryStream(slice);
        using var reader = new BinaryReader(ms);
        var result = new float[n];
        var qOps = QuantizationFactory.Create(HardwareTier.Scalar);
        qOps.ReadFor(dtype, reader, result, n);
        return result;
    }
}