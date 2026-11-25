using System;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// Q4_K normally uses its packed kernel. The generic byte kernel can run the
/// same tensor at a larger repack; these tests pin the opt-in gate and check the
/// two paths agree, including through a real InferenceLinearLayer.
/// </summary>
[Collection("Non-Parallel")]
public sealed class Q4KByteModeTests
{
    private const int SuperBlock = 256;
    private const int RawBlockBytes = 144;
    private const long PackedBlockBytes = 1280;
    private const long ByteBlockBytes = 2304;

    [Fact]
    public void ByteModeIsOptIn()
    {
        bool original = WideByteWeights.UseForQ4K;
        try
        {
            WideByteWeights.UseForQ4K = false;
            Assert.True(WideByteWeights.IsSupported(QuantDType.Q4_K));
            Assert.False(WideByteWeights.Allowed(QuantDType.Q4_K));

            WideByteWeights.UseForQ4K = true;
            Assert.True(WideByteWeights.Allowed(QuantDType.Q4_K));
            Assert.True(WideByteWeights.Allowed(QuantDType.Q4_K_S));
            Assert.True(WideByteWeights.Allowed(QuantDType.Q4_K_M));
        }
        finally
        {
            WideByteWeights.UseForQ4K = original;
        }
    }

    [Fact]
    public unsafe void ByteAndPackedAgreeOnTheSameTensor()
    {
        const int k = 256, n = 16, m = 2;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        long budget = WideRepackBudget.MaxBytes;
        WideRepackBudget.ResetForTests();
        try
        {
            var rng = new Random(4101);
            var raw = RandomQ4K(k, n, rng);
            var input = RandomInput(m * k, rng);

            var packed = Q4KWideWeights.TryCreate(raw, k, n)!;
            var byteWide = WideByteWeights.TryCreate(raw, k, n, QuantDType.Q4_K)!;
            int groups = n / Q4KWideWeights.RowsPerGroup;
            Assert.Equal(groups * (PackedBlockBytes + ByteBlockBytes),
                WideRepackBudget.AllocatedBytes);

            var packedOut = new float[m * n];
            var byteOut = new float[m * n];
            fixed (float* pIn = input)
            fixed (float* pPacked = packedOut)
            fixed (float* pByte = byteOut)
            {
                packed.MatMul(pIn, pPacked, m);
                byteWide.MatMul(pIn, pByte, m);
            }

            for (int i = 0; i < packedOut.Length; i++)
            {
                float want = packedOut[i];
                float got = byteOut[i];
                Assert.True(
                    Math.Abs(got - want) <= 2e-3f * Math.Max(1f, Math.Abs(want)) + 0.5f,
                    $"[{i}] byte={got}, packed={want}");
            }

            packed.Dispose();
            byteWide.Dispose();
        }
        finally
        {
            WideRepackBudget.MaxBytes = budget;
            WideRepackBudget.ResetForTests();
        }
    }

    [Fact]
    public unsafe void LayerSwitchesToByteModeWhenOptedIn()
    {
        const int k = 256, n = 16;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        bool original = WideByteWeights.UseForQ4K;
        long budget = WideRepackBudget.MaxBytes;
        WideRepackBudget.ResetForTests();
        try
        {
            var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);
            var layer = (InferenceLinearLayer)LinearLayerFactory.Create(
                "q4", k, n, false, null, null, QuantDType.Q4_K, mapping);
            Assert.True(layer.WideAllowed);

            var raw = RandomQ4K(k, n, new Random(4102));
            layer.SetRawWeight(raw);
            var input = RandomInput(k, new Random(4103));
            using var inputTensor = new Tensor<float>(1, k);
            input.CopyTo(inputTensor.Data);

            float[] packed;
            using (var output = layer.Forward(inputTensor))
                packed = output.Data.ToArray();

            Assert.True(layer.HasCachedQ4KWide);
            Assert.False(layer.HasCachedWideByte);

            WideByteWeights.UseForQ4K = true;
            layer.SetRawWeight(raw);
            float[] byteMode;
            using (var output = layer.Forward(inputTensor))
                byteMode = output.Data.ToArray();

            Assert.True(layer.HasCachedWideByte);
            Assert.False(layer.HasCachedQ4KWide);

            for (int i = 0; i < packed.Length; i++)
            {
                Assert.True(
                    Math.Abs(byteMode[i] - packed[i]) <=
                        2e-3f * Math.Max(1f, Math.Abs(packed[i])) + 0.5f,
                    $"[{i}] byte={byteMode[i]}, packed={packed[i]}");
            }

            layer.Dispose();
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);
        }
        finally
        {
            WideByteWeights.UseForQ4K = original;
            WideRepackBudget.MaxBytes = budget;
            WideRepackBudget.ResetForTests();
        }
    }

    private static byte[] RandomQ4K(int k, int n, Random rng)
    {
        int blocks = k / SuperBlock;
        var raw = new byte[(long)n * blocks * RawBlockBytes];
        rng.NextBytes(raw);
        for (int off = 0; off < raw.Length; off += RawBlockBytes)
        {
            ushort d = BitConverter.HalfToUInt16Bits((Half)(0.25 + rng.NextDouble()));
            ushort dmin = BitConverter.HalfToUInt16Bits((Half)(0.05 + rng.NextDouble() * 0.1));
            raw[off] = (byte)d;
            raw[off + 1] = (byte)(d >> 8);
            raw[off + 2] = (byte)dmin;
            raw[off + 3] = (byte)(dmin >> 8);
        }
        return raw;
    }

    private static float[] RandomInput(int length, Random rng)
    {
        var x = new float[length];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 4 - 2);
        return x;
    }
}
