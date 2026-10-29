using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Layers;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// Diagnostic: managed allocation of one real <see cref="InferenceLinearLayer.Forward"/>
/// at the Qwen2-0.5B FFN decode shape, built through the same factory and mapping the
/// probe uses. Records the figure; asserts only that it is non-negative.
/// </summary>
public class LayerForwardAllocationTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(896, 4864)]
    [InlineData(4864, 896)]
    public unsafe void RealLayerForward_AllocationPerCall(int k, int n)
    {
        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: true);
        var layer = (InferenceLinearLayer)LinearLayerFactory.Create("probe", k, n, false, null, null, QuantDType.Q8_0, mapping);
        int blocks = k / 32;
        var raw = new byte[(long)n * blocks * 34];
        new Random(5).NextBytes(raw);
        for (long i = 0; i < (long)n * blocks; i++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)0.01f);
            raw[i * 34] = (byte)bits;
            raw[i * 34 + 1] = (byte)(bits >> 8);
        }
        layer.SetRawWeight(raw);

        using var input = new Tensor<float>(1, k);
        for (int i = 0; i < k; i++) input.Data[i] = (float)(i % 7) * 0.1f;

        for (int i = 0; i < 50; i++) layer.Forward(input).Dispose();

        const int iters = 1000;
        long before = GC.GetTotalAllocatedBytes(precise: true);
        for (int i = 0; i < iters; i++) layer.Forward(input).Dispose();
        long after = GC.GetTotalAllocatedBytes(precise: true);

        output.WriteLine($"Layer K={k} N={n} (wide={layer.WideAllowed}): {(after - before) / (double)iters:F0} B/forward incl. result tensor");
        layer.Dispose();
        Assert.True(after >= before);
    }
}
