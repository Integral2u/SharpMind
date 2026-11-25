using System;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// The wide repacks are guarded by one process-wide byte budget rather than a
/// per-layer cap, and a complete repack lets the layer drop its raw bytes. These
/// tests pin both behaviours: a reservation that would exceed the cap is refused,
/// releasing returns the bytes, and a wide-only layer keeps computing and reloads
/// correctly after the raw tensor is gone.
/// </summary>
[Collection("Non-Parallel")]
public sealed class WideRepackBudgetTests
{
    private const int SuperBlock = 256;
    private const int RawBlockBytes = 144;
    private const long Q4KBlockBytes = 1536; // eight rows × eight 32-value sub-blocks × 192 B

    [Fact]
    public void BudgetRefusesBeyondTheCapAndReleaseAllowsRetry()
    {
        long original = WideRepackBudget.MaxBytes;
        WideRepackBudget.ResetForTests();
        try
        {
            WideRepackBudget.MaxBytes = 1_000;

            Assert.True(WideRepackBudget.TryReserve(600));
            Assert.Equal(600, WideRepackBudget.AllocatedBytes);

            Assert.False(WideRepackBudget.TryReserve(600));
            Assert.Equal(600, WideRepackBudget.AllocatedBytes);

            WideRepackBudget.Release(600);
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);
            Assert.True(WideRepackBudget.TryReserve(1_000));
        }
        finally
        {
            WideRepackBudget.MaxBytes = original;
            WideRepackBudget.ResetForTests();
        }
    }

    [Fact]
    public void Q4KRepackIsRefusedWhileTheBudgetIsExhausted()
    {
        if (!Q4KWideWeights.IsSupported(SuperBlock, 8)) return;

        long original = WideRepackBudget.MaxBytes;
        WideRepackBudget.ResetForTests();
        try
        {
            var raw = RandomQ4K(SuperBlock, 8, new Random(1));
            WideRepackBudget.MaxBytes = Q4KBlockBytes;

            var first = Q4KWideWeights.TryCreate(raw, SuperBlock, 8);
            Assert.NotNull(first);
            Assert.Equal(Q4KBlockBytes, WideRepackBudget.AllocatedBytes);

            Assert.Null(Q4KWideWeights.TryCreate(raw, SuperBlock, 8));

            first!.Dispose();
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);

            var second = Q4KWideWeights.TryCreate(raw, SuperBlock, 8);
            Assert.NotNull(second);
            second!.Dispose();
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);
        }
        finally
        {
            WideRepackBudget.MaxBytes = original;
            WideRepackBudget.ResetForTests();
        }
    }

    [Fact]
    public unsafe void CompleteRepackDropsRawAndKeepsComputing()
    {
        const int k = 256, n = 16;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        long original = WideRepackBudget.MaxBytes;
        WideRepackBudget.ResetForTests();
        try
        {
            var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);
            var layer = (InferenceLinearLayer)LinearLayerFactory.Create(
                "q4", k, n, false, null, null, QuantDType.Q4_K, mapping);
            Assert.True(layer.WideAllowed);

            var raw = RandomQ4K(k, n, new Random(42));
            layer.SetRawWeight(raw);

            var input = RandomInput(k, new Random(43));
            using var inputTensor = new Tensor<float>(1, k);
            input.CopyTo(inputTensor.Data);

            float[] first;
            using (var output = layer.Forward(inputTensor))
                first = output.Data.ToArray();

            Assert.Null(layer.RawQuantizedData);
            Assert.True(layer.HasCachedQ4KWide);
            Assert.Equal(2 * Q4KBlockBytes, WideRepackBudget.AllocatedBytes);

            // A second forward must take the wide-only path, not the raw fallback.
            using (var output = layer.Forward(inputTensor))
                Assert.Equal(first, output.Data.ToArray());

            // A reload installs new raw bytes and the next forward repacks and drops them.
            var raw2 = RandomQ4K(k, n, new Random(44));
            layer.SetRawWeight(raw2);
            Assert.Same(raw2, layer.RawQuantizedData);
            Assert.False(layer.HasCachedQ4KWide);
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);

            using (var output = layer.Forward(inputTensor))
            {
                Assert.Equal(n, output.ElementCount);
                Assert.NotEqual(first, output.Data.ToArray());
            }

            Assert.Null(layer.RawQuantizedData);
            Assert.True(layer.HasCachedQ4KWide);

            // Freeing a wide-only layer releases the repack and its budget.
            layer.FreeFloatWeight();
            Assert.False(layer.HasCachedQ4KWide);
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);

            // The layer is still usable after a reload.
            layer.SetRawWeight(raw);
            using (var output = layer.Forward(inputTensor))
                Assert.Equal(first, output.Data.ToArray());

            layer.Dispose();
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);
        }
        finally
        {
            WideRepackBudget.MaxBytes = original;
            WideRepackBudget.ResetForTests();
        }
    }

    [Fact]
    public void LogitOpsDisposeReleasesItsRepack()
    {
        const int k = 256, n = 16;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        long original = WideRepackBudget.MaxBytes;
        WideRepackBudget.ResetForTests();
        try
        {
            var raw = RandomQ4K(k, n, new Random(55));
            var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);
            var ops = LogitOpsFactory.Create(null, raw, QuantDType.Q4_K, mapping);
            Assert.True(ops.WideAllowed);

            var input = RandomInput(k, new Random(56));
            using var inputTensor = new Tensor<float>(1, k);
            input.CopyTo(inputTensor.Data);
            using (ops.Project(inputTensor, 1, k, n))
            {
            }

            Assert.Equal(2 * Q4KBlockBytes, WideRepackBudget.AllocatedBytes);
            ops.Dispose();
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);
        }
        finally
        {
            WideRepackBudget.MaxBytes = original;
            WideRepackBudget.ResetForTests();
        }
    }

    [Fact]
    public unsafe void CompleteRepackReleasesTheOwningBlockRaw()
    {
        const int k = 256, n = 16;
        if (!Q4KWideWeights.IsSupported(k, n)) return;

        long original = WideRepackBudget.MaxBytes;
        WideRepackBudget.ResetForTests();
        try
        {
            var block = new TransformerWeights.BlockWeights
            {
                CanReleaseRawAfterRepack = true,
            };
            block.RawWq = RandomQ4K(k, n, new Random(61));
            block.RawWk = RandomQ4K(k, n, new Random(62));

            var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);
            var layer = (InferenceLinearLayer)LinearLayerFactory.Create(
                "q4", k, n, false, null, null, QuantDType.Q4_K, mapping);
            layer.RawReleaseCallback = block.ReleaseRawSource;
            layer.SetRawWeight(block.RawWq);

            var input = RandomInput(k, new Random(63));
            using var inputTensor = new Tensor<float>(1, k);
            input.CopyTo(inputTensor.Data);

            float[] first;
            using (var output = layer.Forward(inputTensor))
                first = output.Data.ToArray();

            Assert.Null(block.RawWq);
            Assert.NotNull(block.RawWk);
            Assert.True(block.RawWasReleased);
            Assert.Null(layer.RawQuantizedData);

            using (var output = layer.Forward(inputTensor))
                Assert.Equal(first, output.Data.ToArray());

            layer.Dispose();
            Assert.Equal(0, WideRepackBudget.AllocatedBytes);
        }
        finally
        {
            WideRepackBudget.MaxBytes = original;
            WideRepackBudget.ResetForTests();
        }
    }

    [Fact]
    public void RawReleaseRespectsTheOptInAndStreamingGuard()
    {
        var raw = RandomQ4K(SuperBlock, 8, new Random(64));
        var block = new TransformerWeights.BlockWeights();
        block.RawWq = raw;

        block.ReleaseRawSource(raw);
        Assert.Same(raw, block.RawWq);
        Assert.False(block.RawWasReleased);

        block.CanReleaseRawAfterRepack = true;
        block.BufferAllocator = _ => new byte[1];
        block.ReleaseRawSource(raw);
        Assert.Same(raw, block.RawWq);
        Assert.False(block.RawWasReleased);

        block.BufferAllocator = null;
        block.ReleaseRawSource(raw);
        Assert.Null(block.RawWq);
        Assert.True(block.RawWasReleased);
    }

    [Fact]
    public void RawReleaseRemovesOnlyTheMatchingExpertPlane()
    {
        var plane3 = RandomQ4K(SuperBlock, 8, new Random(65));
        var plane4 = RandomQ4K(SuperBlock, 8, new Random(66));
        var block = new TransformerWeights.BlockWeights
        {
            CanReleaseRawAfterRepack = true,
            RawWgateExp = new() { [3] = plane3, [4] = plane4 },
        };

        block.ReleaseRawSource(plane3);

        Assert.False(block.RawWgateExp.ContainsKey(3));
        Assert.True(block.RawWgateExp.ContainsKey(4));
        Assert.Same(plane4, block.RawWgateExp[4]);
        Assert.True(block.RawWasReleased);
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
