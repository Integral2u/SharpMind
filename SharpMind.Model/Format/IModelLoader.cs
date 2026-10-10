using SharpMind.Core.Quantization;

namespace SharpMind.Model.Format;

public interface IModelLoader
{
    /// <summary>
    /// Full dequantization pass — reads every tensor from the file, dequantizes
    /// to float, and fills the target float tensors. Also populates raw quantized
    /// data and tensor metadata on block weights.
    /// </summary>
    void LoadAllWeights(TransformerWeights weights, IProgress<float>? progress = null, CancellationToken? cancellationToken = null);

    /// <summary>
    /// Loads and dequantizes tensors for a single transformer block layer.
    /// Reads raw quantized data, populates tensor metadata, and dequantizes
    /// to float for the specified layer index.
    /// </summary>
    void LoadLayerWeights(int layerIndex, TransformerWeights weights, CancellationToken? cancellationToken = null);

    /// <summary>
    /// Whether this loader can fetch individual routed experts (plane-sliced fused tensors
    /// or per-expert named tensors). Streaming MoE expert residency needs this: without it the
    /// <c>DesiredResidentExperts</c> option is ignored and layers load wholesale as before.
    /// </summary>
    bool SupportsExpertSlicing => false;

    /// <summary>
    /// Loads a layer's tensors excluding the routed-expert planes. Residency keeps the pinned
    /// experts from a per-layer store, so the layer load must not re-read every expert. The
    /// default falls back to a whole-layer load (no slicing support).
    /// </summary>
    void LoadLayerNonExpertWeights(int layerIndex, TransformerWeights weights, CancellationToken? cancellationToken = null)
        => LoadLayerWeights(layerIndex, weights, cancellationToken);

    /// <summary>
    /// Loads the raw quantized planes for the given routed experts only. Used by streaming MoE
    /// residency on demand when the router selects an expert that is not resident. The default
    /// is a no-op (no slicing support).
    /// </summary>
    void LoadExpertWeights(int layerIndex, IReadOnlyList<int> expertIndices, TransformerWeights weights, CancellationToken? cancellationToken = null) { }

    /// <summary>
    /// Loads global (non-block) tensors: embedding weight, final norm weight,
    /// and LM head weight. Called once during <see cref="TransformerWeights.InitializeWeights"/>
    /// in streaming mode; block-level tensors are loaded per-layer by
    /// <see cref="LoadLayerWeights"/>.
    /// </summary>
    void LoadGlobalTensors(TransformerWeights weights, CancellationToken? cancellationToken = null);
}
