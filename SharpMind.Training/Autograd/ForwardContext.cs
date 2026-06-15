using SharpMind.Core.Tensors;

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

    /// <summary>Final norm output [Batch, SeqLen, HiddenDim].</summary>
    public Tensor<float>? FinalNormOut { get; set; }

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
