using System;
using System.Collections.Generic;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// The generic byte repack normalises Q4_0/Q4_1/Q5_0/Q5_1/IQ4_NL into
/// scale * dot(value, activation) - bias * sum(activation). Checked against a
/// double-precision reference of exactly that arithmetic, then through the layer
/// and LogitOps integration.
/// </summary>
public sealed class WideByteWeightsTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var dtype in new[]
                 {
                     QuantDType.Q4_0, QuantDType.Q4_1, QuantDType.Q5_0,
                     QuantDType.Q5_1, QuantDType.IQ4_NL,
                 })
            foreach (int k in new[] { 32, 64, 128 })
                foreach (int n in new[] { 8, 16, 24 })
                    foreach (int m in new[] { 1, 3, 16 })
                        yield return new object[] { dtype, k, n, m };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public unsafe void MatchesGroupedReference(QuantDType dtype, int k, int n, int m)
    {
        if (!WideByteWeights.IsSupported(dtype)) return;

        var rng = new Random(5001 + (int)dtype * 31 + k * 7 + n * 13 + m);
        var raw = RandomRaw(dtype, k, n, rng);
        var input = RandomInput(m * k, rng);
        var wide = WideByteWeights.TryCreate(raw, k, n, dtype);
        Assert.NotNull(wide);

        var actual = new float[m * n];
        Array.Fill(actual, float.NaN);
        fixed (float* pIn = input)
        fixed (float* pOut = actual)
            wide.MatMul(pIn, pOut, m);

        for (int p = 0; p < m; p++)
        {
            var (q, dq) = QuantizeReference(input, p * k, k / 32);
            for (int row = 0; row < n; row++)
            {
                double expected = ReferenceRow(dtype, raw, row, k, q, dq);
                float got = actual[p * n + row];
                string at = $"{dtype} K={k} N={n} M={m} [{p},{row}]";
                Assert.False(float.IsNaN(got), $"{at}: unwritten");
                Assert.True(
                    Math.Abs(got - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)) + 1e-2,
                    $"{at}: wide={got}, reference={expected}");
            }
        }
    }

    public static IEnumerable<object[]> OtherCases()
    {
        foreach (var dtype in new[]
                 {
                     QuantDType.Q5_K, QuantDType.Q2_K, QuantDType.Q3_K,
                     QuantDType.IQ4_XS, QuantDType.IQ3_XXS, QuantDType.IQ2_XXS,
                     QuantDType.IQ2_XS, QuantDType.IQ2_S, QuantDType.IQ3_S,
                 })
            foreach (int k in new[] { 256, 512 })
                foreach (int n in new[] { 8, 16 })
                    foreach (int m in new[] { 1, 3 })
                        yield return new object[] { dtype, k, n, m };

        foreach (int k in new[] { 32, 64 })
            foreach (int n in new[] { 8, 16 })
                foreach (int m in new[] { 1, 3 })
                    yield return new object[] { QuantDType.MXFP4, k, n, m };
    }

    [Theory]
    [MemberData(nameof(OtherCases))]
    public unsafe void MatchesScalarKernels(QuantDType dtype, int k, int n, int m)
    {
        if (!WideByteWeights.IsSupported(dtype)) return;

        var rng = new Random(6101 + (int)dtype * 17 + k * 11 + n * 5 + m);
        var raw = RandomRaw(dtype, k, n, rng);
        var input = RandomInput(m * k, rng);
        var wide = WideByteWeights.TryCreate(raw, k, n, dtype);
        Assert.NotNull(wide);

        var actual = new float[m * n];
        var expected = new float[m * n];
        var dequantized = new float[m * k];
        for (int p = 0; p < m; p++)
        {
            var (q, dq) = QuantizeReference(input, p * k, k / 32);
            for (int b = 0; b < k / 32; b++)
                for (int i = 0; i < 32; i++)
                    dequantized[p * k + b * 32 + i] = q[b * 32 + i] * dq[b];
        }

        Array.Fill(actual, float.NaN);
        fixed (float* pIn = input)
        fixed (float* pQ = dequantized)
        fixed (float* pOut = actual)
        fixed (float* pRef = expected)
        fixed (byte* pRaw = raw)
        {
            wide.MatMul(pIn, pOut, m);
            switch (dtype)
            {
                case QuantDType.Q5_K:
                    QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.Q2_K:
                    QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.Q3_K:
                    QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.IQ4_XS:
                    QuantizationKernels.QuantizedMatMulIQ4_XS_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.IQ3_XXS:
                    QuantizationKernels.QuantizedMatMulIQ3_XXS_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.IQ2_XXS:
                    QuantizationKernels.QuantizedMatMulIQ2_XXS_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.IQ2_XS:
                    QuantizationKernels.QuantizedMatMulIQ2_XS_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.IQ2_S:
                    QuantizationKernels.QuantizedMatMulIQ2_S_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.IQ3_S:
                    QuantizationKernels.QuantizedMatMulIQ3_S_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
                case QuantDType.MXFP4:
                    QuantizationKernels.QuantizedMatMulMXFP4_Serial_Scalar(pQ, pRaw, pRef, m, k, n);
                    break;
            }
        }

        for (int i = 0; i < actual.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), $"{dtype} [{i}]: unwritten");
            Assert.True(
                Math.Abs(actual[i] - expected[i]) <= 1e-3f * Math.Max(1f, Math.Abs(expected[i])) + 0.05f,
                $"{dtype} [{i}]: wide={actual[i]}, scalar={expected[i]}");
        }
    }

    [Fact]
    public void UnsupportedShapesAreRefused()
    {
        Assert.False(WideByteWeights.IsSupported(QuantDType.Q8_0));
        Assert.Null(WideByteWeights.TryCreate(new byte[16], 32, 7, QuantDType.Q4_0));
        Assert.Null(WideByteWeights.TryCreate(new byte[17], 32, 8, QuantDType.Q4_0));
        Assert.Null(WideByteWeights.TryCreate(new byte[16], 48, 8, QuantDType.Q4_0));
    }

    [Fact]
    public unsafe void LayerAndLogitOpsUseTheRepack()
    {
        const int k = 64, n = 16;
        var rng = new Random(77);
        var raw = RandomRaw(QuantDType.Q4_0, k, n, rng);
        var input = RandomInput(k, rng);
        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);

        var layer = (InferenceLinearLayer)LinearLayerFactory.Create(
            "q4", k, n, false, null, null, QuantDType.Q4_0, mapping);
        Assert.True(layer.WideAllowed);
        layer.SetRawWeight(raw);
        using var inputTensor = new Tensor<float>(1, k);
        input.CopyTo(inputTensor.Data);
        using (layer.Forward(inputTensor))
        {
        }
        Assert.True(layer.HasCachedWideByte);
        layer.FreeFloatWeight();
        Assert.False(layer.HasCachedWideByte);
        layer.Dispose();

        var ops = LogitOpsFactory.Create(null, raw, QuantDType.Q4_0, mapping);
        Assert.True(ops.WideAllowed);
        using var projected = ops.Project(inputTensor, 1, k, n);

        var wide = WideByteWeights.TryCreate(raw, k, n, QuantDType.Q4_0)!;
        var reference = new float[n];
        fixed (float* pIn = input)
        fixed (float* pRef = reference)
            wide.MatMul(pIn, pRef, 1);
        Assert.Equal(reference, projected.Data.ToArray());
    }

    private static byte[] RandomRaw(QuantDType dtype, int k, int n, Random rng)
    {
        (int blockSize, int blockBytes) = dtype switch
        {
            QuantDType.Q4_0 => (32, 18),
            QuantDType.Q4_1 => (32, 20),
            QuantDType.Q5_0 => (32, 22),
            QuantDType.Q5_1 => (32, 24),
            QuantDType.IQ4_NL => (32, 18),
            QuantDType.MXFP4 => (32, 17),
            QuantDType.Q5_K => (256, 176),
            QuantDType.Q2_K => (256, 84),
            QuantDType.Q3_K => (256, 110),
            QuantDType.IQ4_XS => (256, 136),
            QuantDType.IQ3_XXS => (256, 98),
            QuantDType.IQ2_XXS => (256, 66),
            QuantDType.IQ2_XS => (256, 74),
            QuantDType.IQ2_S => (256, 82),
            QuantDType.IQ3_S => (256, 110),
            _ => throw new NotSupportedException(dtype.ToString()),
        };
        int blocks = k / blockSize;
        var raw = new byte[(long)n * blocks * blockBytes];
        rng.NextBytes(raw);
        for (int off = 0; off < raw.Length; off += blockBytes)
        {
            if (dtype == QuantDType.MXFP4)
            {
                raw[off] = (byte)(120 + rng.Next(8)); // E8M0 exponent, never 255
                continue;
            }

            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.1 + rng.NextDouble()));
            int dOffset = dtype is QuantDType.Q2_K ? 80 : dtype is QuantDType.Q3_K ? 108 : 0;
            raw[off + dOffset] = (byte)d;
            raw[off + dOffset + 1] = (byte)(d >> 8);

            int minOffset = dtype switch
            {
                QuantDType.Q4_1 or QuantDType.Q5_1 => 2,
                QuantDType.Q5_K => 2,
                QuantDType.Q2_K => 82,
                _ => -1,
            };
            if (minOffset >= 0)
            {
                ushort m = BitConverter.HalfToUInt16Bits((Half)(-0.2 + rng.NextDouble() * 0.4));
                raw[off + minOffset] = (byte)m;
                raw[off + minOffset + 1] = (byte)(m >> 8);
            }
        }
        return raw;
    }

    private static float[] RandomInput(int length, Random rng)
    {
        var x = new float[length];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        return x;
    }

    private static (int[] Q, float[] D) QuantizeReference(float[] input, int offset, int blocks32)
    {
        var q = new int[blocks32 * 32];
        var d = new float[blocks32];
        for (int b = 0; b < blocks32; b++)
        {
            float amax = 0;
            for (int j = 0; j < 32; j++) amax = MathF.Max(amax, MathF.Abs(input[offset + b * 32 + j]));
            d[b] = amax / 127f;
            float id = amax >= 1e-30f ? 127f / amax : 0f;
            for (int j = 0; j < 32; j++)
                q[b * 32 + j] = (int)MathF.Round(
                    input[offset + b * 32 + j] * id, MidpointRounding.ToEven);
        }
        return (q, d);
    }

    private static unsafe double ReferenceRow(QuantDType dtype, byte[] raw, int row, int k,
        int[] q, float[] dq)
    {
        int blockBytes = dtype switch
        {
            QuantDType.Q4_0 => 18,
            QuantDType.Q4_1 => 20,
            QuantDType.Q5_0 => 22,
            QuantDType.Q5_1 => 24,
            QuantDType.IQ4_NL => 18,
            _ => throw new NotSupportedException(dtype.ToString()),
        };
        int blocks = k / 32;
        double sum = 0;
        fixed (byte* pRaw = raw)
        {
            for (int b = 0; b < blocks; b++)
            {
                byte* block = pRaw + ((long)row * blocks + b) * blockBytes;
                float d = (float)BitConverter.UInt16BitsToHalf(UnsafeReadUInt16(block));
                float m = dtype is QuantDType.Q4_1 or QuantDType.Q5_1
                    ? (float)BitConverter.UInt16BitsToHalf(UnsafeReadUInt16(block + 2))
                    : 0;
                long dot = 0, sumQ = 0;
                for (int i = 0; i < 32; i++)
                {
                    int value = dtype switch
                    {
                        QuantDType.Q4_0 => Nibble(block, 2, i) - 8,
                        QuantDType.Q4_1 => Nibble(block, 4, i),
                        QuantDType.Q5_0 => Nibble(block, 6, i)
                            | (int)((UnsafeReadUInt32(block + 2) >> i) & 1) << 4,
                        QuantDType.Q5_1 => Nibble(block, 8, i)
                            | (int)((UnsafeReadUInt32(block + 4) >> i) & 1) << 4,
                        QuantDType.IQ4_NL => Iq4Nl[Nibble(block, 2, i)],
                        _ => throw new NotSupportedException(),
                    };
                    if (dtype == QuantDType.Q5_0) value -= 16;

                    int activation = q[b * 32 + i];
                    dot += value * activation;
                    sumQ += activation;
                }
                sum += dq[b] * (d * dot + m * sumQ);
            }
        }
        return sum;
    }

    private static unsafe int Nibble(byte* block, int offset, int i)
        => i < 16 ? (block[offset + i] & 0x0F) : (block[offset + i - 16] >> 4);

    private static unsafe uint UnsafeReadUInt32(byte* p)
        => (uint)(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));

    private static unsafe ushort UnsafeReadUInt16(byte* p)
        => (ushort)(p[0] | (p[1] << 8));

    private static readonly int[] Iq4Nl =
        [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
}
