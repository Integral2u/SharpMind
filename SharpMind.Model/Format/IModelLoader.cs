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

    /// <summary>
    /// Loads and dequantizes tensors for a single transformer block layer.
    /// Reads raw quantized data, populates tensor metadata, and dequantizes
    /// to float for the specified layer index.
    /// </summary>
    void LoadLayerWeights(int layerIndex, TransformerWeights weights);

    /// <summary>
    /// Loads global (non-block) tensors: embedding weight, final norm weight,
    /// and LM head weight. Called once during <see cref="TransformerWeights.InitializeWeights"/>
    /// in streaming mode; block-level tensors are loaded per-layer by
    /// <see cref="LoadLayerWeights"/>.
    /// </summary>
    void LoadGlobalTensors(TransformerWeights weights);
}
