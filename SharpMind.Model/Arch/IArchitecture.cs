using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Layers;

namespace SharpMind.Model.Arch;

/// <summary>
/// Defines how transformer blocks are connected and what masking is applied.
/// Built-in implementations: <see cref="DecoderArch"/>, <see cref="EncoderArch"/>.
/// Implement this interface to add encoder-decoder (T5/BART) or other architectures
/// without modifying the core layers.
/// </summary>
public interface IArchitecture : IDisposable
{
    public Tensor<float> Forward(Tensor<float> hiddenStates, int positionOffset = 0, SharpMind.Core.Memory.IWorkspace? workspace = null);
    public Tensor<float> Forward(Tensor<float> hiddenStates, IKVCache[] caches, int positionOffset = 0, SharpMind.Core.Memory.IWorkspace? workspace = null);
    public int NumLayers { get; }
    public IEnumerable<Parameter> Parameters();

    public void Backward(Tensor<float> dOutput);

    /// <summary>The transformer blocks this architecture runs, in order. Streaming
    /// weight loading pushes per-layer weights through these; the default is empty
    /// for architectures that do not participate.</summary>
    public TransformerBlock[] Blocks => [];

    /// <summary>Optional callback invoked before each block during Forward. Streaming
    /// load mode wires it to load the incoming layer and free the outgoing ones.</summary>
    public Action<int>? BeforeBlock { get => null; set { } }
}
