using SharpMind.Core;
using SharpMind.Core.Quantization;
using SharpMind.Model.Config;
using SharpMind.Model.Layers;
using Xunit;

namespace SharpMind.Tests.Model;

/// <summary>
/// An InferenceLinearLayer runs the quantized forward off RawQuantizedData and
/// never reads its float _weight. In quantized-resident loading the caller
/// passes weight: null, and the base LinearLayer constructor used to fill that
/// null with a full [inFeatures, outFeatures] F32 tensor — a second dead copy
/// of every weight, on top of the quantized bytes. Tolerable on a 0.5B model,
/// but ~28 GB on a 7B and an out-of-memory at load on a 14B, allocated during
/// CreateTransformer before anything could free it.
///
/// A quantized inference layer built without a float weight must hold only a
/// placeholder, not an in×out tensor.
/// </summary>
public sealed class QuantizedResidentWeightTests
{
    [Theory]
    [InlineData(QuantDType.Q8_0)]
    [InlineData(QuantDType.Q4_K)]
    public void QuantizedInferenceLayer_WithNullWeight_AllocatesNoFullFloatWeight(QuantDType dtype)
    {
        const int inFeatures = 512;
        const int outFeatures = 4096;   // in×out = 2,097,152 floats if the bug is present
        var mapping = SharpMindConfig.Gpt.ToJigSawMapping(parallel: false);

        var layer = LinearLayerFactory.Create(
            "ffn_test", inFeatures, outFeatures, bias: false,
            weight: null, biasTensor: null, dtype, mapping);

        // The forward reads RawQuantizedData, so the float weight is dead; it must
        // not be materialised at full size. Anything on the order of one column is
        // fine (the FreeFloatWeight placeholder is [inFeatures, 1]).
        Assert.True(layer.Weight.ElementCount <= inFeatures,
            $"float weight has {layer.Weight.ElementCount} elements; a quantized-resident " +
            $"layer must not allocate the full {(long)inFeatures * outFeatures}");
    }
}
