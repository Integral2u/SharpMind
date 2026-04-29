using SharpMind.Core.Tensors;

namespace SharpMind.Model.Arch;

/// <summary>
/// Defines how transformer blocks are connected and what masking is applied.
/// Built-in implementations: <see cref="DecoderArch"/>, <see cref="EncoderArch"/>.
/// Implement this interface to add encoder-decoder (T5/BART) or other architectures
/// without modifying the core layers.
/// </summary>
public interface IArchitecture : IDisposable
{
    /// <summary>
    /// Runs the full forward pass through all transformer blocks.
    /// Input:  [Batch, SeqLen, HiddenDim]
    /// Output: [Batch, SeqLen, HiddenDim]
    /// </summary>
    Tensor<float> Forward(Tensor<float> hiddenStates, int positionOffset = 0);

    /// <summary>Number of transformer blocks.</summary>
    int NumLayers { get; }
}
