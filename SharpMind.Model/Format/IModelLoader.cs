using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

public interface IModelLoader
{
    /// <summary>
    /// Records tensor metadata (offset, size, QuantDType) into
    /// <see cref="TransformerWeights.BlockWeights.TensorMeta"/> and top-level
    /// raw data for embedding/lm_head tensors.
    /// Called by both <see cref="TransformerWeightsFull.InitializeWeights"/>
    /// and <see cref="TransformerWeightsCached.InitializeWeights"/>.
    /// </summary>
    void PreInit(TransformerWeights weights, IProgress<float>? progress = null);

    /// <summary>
    /// Full dequantization pass — reads every tensor from the file, dequantizes
    /// to float, and fills the target float tensors. Also populates raw quantized
    /// data on block weights.
    /// Called by <see cref="TransformerWeightsFull.InitializeWeights"/>.
    /// </summary>
    void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null);

    /// <summary>
    /// Loads and dequantizes a single layer's tensors (reads raw bytes from file,
    /// sets raw data + QuantDType on the block, and fills float tensors).
    /// Called by <see cref="TransformerWeightsCached.LoadLayer"/>.
    /// </summary>
    void LoadLayer(TransformerWeights weights, int layerIndex);

    /// <summary>
    /// Loads top-level (non-block) tensors — embedding weight and lm_head weight —
    /// from the model file. Sets both raw quantized data and dequantized float tensors.
    /// Called by <see cref="TransformerWeightsCached.InitializeWeights"/>.
    /// </summary>
    void LoadTopLevelTensors(TransformerWeights weights);
}
