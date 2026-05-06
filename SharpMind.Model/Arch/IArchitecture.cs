using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Arch;

/// <summary>
/// Defines how transformer blocks are connected and what masking is applied.
/// Built-in implementations: <see cref="DecoderArch"/>, <see cref="EncoderArch"/>.
/// Implement this interface to add encoder-decoder (T5/BART) or other architectures
/// without modifying the core layers.
/// </summary>
public interface IArchitecture : IDisposable
{
    Tensor<float> Forward(Tensor<float> hiddenStates, int positionOffset = 0);
    Tensor<float> Forward(Tensor<float> hiddenStates, KVCache[] caches, int positionOffset = 0);
    int NumLayers { get; }
    IEnumerable<Parameter> Parameters();
    
    void Backward(Tensor<float> dOutput);
}
