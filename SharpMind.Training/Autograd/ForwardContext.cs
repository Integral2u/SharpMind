using SharpMind.Core.Tensors;
using SharpMind.Model.Layers;

namespace SharpMind.Training.Autograd;

/// <summary>
/// Stores the intermediate activations produced during a training forward pass.
/// The backward pass reads these rather than recomputing them.
///
/// One <see cref="BlockContext"/> is created per transformer block.
/// The whole context is disposed after the optimizer step to release memory.
/// </summary>
public sealed class ForwardContext : IDisposable
{
    private bool _disposed;

    // Top-level

    /// <summary>Token IDs used in this forward pass — needed for embedding backward.</summary>
    public Tensor<int>? TokenIds { get; set; }

    /// <summary>Embedding output [Batch, SeqLen, HiddenDim].</summary>
    public Tensor<float>? EmbeddingOut { get; set; }

    /// <summary>
    /// True when the forward pass added learned positional embeddings to the
    /// token embeddings in place; the backward pass must then accumulate the
    /// gradient into the position-embedding parameter rows.
    /// </summary>
    public bool UsesLearnedPositions { get; set; }

    /// <summary>Batch size of the forward pass (see <see cref="UsesLearnedPositions"/>).</summary>
    public int Batch { get; set; }

    /// <summary>Sequence length of the forward pass (see <see cref="UsesLearnedPositions"/>).</summary>
    public int SeqLen { get; set; }

    /// <summary>Final norm output [Batch, SeqLen, HiddenDim].</summary>
    public Tensor<float>? FinalNormOut { get; set; }

    /// <summary>Final norm input (last block output) [Batch, SeqLen, HiddenDim].</summary>
    public Tensor<float>? FinalNormInput { get; set; }

    /// <summary>Per-row saved state of the final norm (input/output/scalar param snapshots).</summary>
    public NormLayerState? FinalNormState { get; set; }

    /// <summary>Logits [Batch, SeqLen, VocabSize].</summary>
    public Tensor<float>? Logits { get; set; }

    // Per-block contexts

    public List<BlockContext> Blocks { get; } = [];

    // Disposal

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        TokenIds?.Dispose();
        EmbeddingOut?.Dispose();
        FinalNormOut?.Dispose();
        Logits?.Dispose();
        foreach (var b in Blocks) b.Dispose();
    }
}
