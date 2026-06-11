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
    public Tensor<float> Forward(Tensor<float> hiddenStates, int positionOffset = 0, SharpMind.Core.Memory.Workspace? workspace = null);
    public Tensor<float> Forward(Tensor<float> hiddenStates, IKVCache[] caches, int positionOffset = 0, SharpMind.Core.Memory.Workspace? workspace = null);
    public int NumLayers { get; }
    public IEnumerable<Parameter> Parameters();

    public void Backward(Tensor<float> dOutput);
}
