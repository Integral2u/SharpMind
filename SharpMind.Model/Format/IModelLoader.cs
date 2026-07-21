using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

public interface IModelLoader
{
    /// <summary>
    /// Full dequantization pass — reads every tensor from the file, dequantizes
    /// to float, and fills the target float tensors. Also populates raw quantized
    /// data and tensor metadata on block weights.
    /// </summary>
    void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null);
}
