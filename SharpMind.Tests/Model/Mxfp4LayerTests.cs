using System;
using System.IO;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// End-to-end MXFP4: the layer factory has to resolve a compound the dispatch
/// table actually registers, <see cref="InferenceLinearLayer.SetRawWeight"/> has
/// to accept the 17-bytes-per-32 shape, and the forward has to agree with the
/// reader's own dequantization of the same bytes.
///
/// The per-kernel tests cannot catch a missing registration: they call
/// <c>VecDotMXFP4</c> directly, which is the step after the compound lookup that
/// fails at model load. This is the same seam that silently handed IQ1_S/TQ1_0
/// tensors to the F32 kernel.
/// </summary>
public sealed class Mxfp4LayerTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void Forward_MatchesDequantizedReference(int m)
    {
        const int k = 96;   // three 32-value blocks per row
        const int n = 40;

        int rowBytes = (k / 32) * 17;
        var raw = new byte[rowBytes * n];
        new Random(4242).NextBytes(raw);
        for (int row = 0; row < n; row++)
            for (int b = 0; b < k / 32; b++)
                raw[row * rowBytes + b * 17] = 127; // E8M0 2^0

        var weights = new float[k * n];
        using (var reader = new BinaryReader(new MemoryStream(raw, writable: false)))
        {
            QuantizationFactory.Create(HardwareTier.Scalar)
                .ReadFor(QuantDType.MXFP4, reader, weights, weights.Length);
        }

        var input = new float[m * k];
        var rng = new Random(7);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: false);
        using var layer = LinearLayerFactory.Create(
            "mxfp4_proj", k, n, bias: false, weight: null, biasTensor: null, QuantDType.MXFP4, mapping);
        layer.SetRawWeight(raw);

        using var x = new Tensor<float>(m, k);
        input.CopyTo(x.Data);
        using var y = layer.Forward(x);

        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                double expected = 0;
                for (int i = 0; i < k; i++)
                    expected += input[row * k + i] * weights[col * k + i];

                float actual = y.Data[row * n + col];
                Assert.True(Math.Abs(actual - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                    $"row {row} col {col}: expected {expected}, got {actual}");
            }
        }
    }

    /// <summary>
    /// A buffer sized for the flat (SMM-style) layout must be rejected when the
    /// GGUF row-padded count differs — the guard that stopped a mismatched-shape
    /// model from reading past the buffer instead of failing at load.
    /// </summary>
    [Fact]
    public void SetRawWeight_RejectsAWronglySizedBuffer()
    {
        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: false);
        using var layer = LinearLayerFactory.Create(
            "mxfp4_proj", 96, 40, bias: false, weight: null, biasTensor: null, QuantDType.MXFP4, mapping);

        Assert.Throws<NotSupportedException>(() => layer.SetRawWeight(new byte[100]));
    }
}
