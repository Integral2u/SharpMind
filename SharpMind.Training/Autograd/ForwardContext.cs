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

    // ── Top-level ─────────────────────────────────────────────────────────

    /// <summary>Token IDs used in this forward pass — needed for embedding backward.</summary>
    public Tensor<int>? TokenIds { get; set; }

    /// <summary>Embedding output [Batch, SeqLen, HiddenDim].</summary>
    public Tensor<float>? EmbeddingOut { get; set; }

    /// <summary>Final norm output [Batch, SeqLen, HiddenDim].</summary>
    public Tensor<float>? FinalNormOut { get; set; }

    /// <summary>Logits [Batch, SeqLen, VocabSize].</summary>
    public Tensor<float>? Logits { get; set; }

    // ── Per-block contexts ────────────────────────────────────────────────

    public List<BlockContext> Blocks { get; } = [];

    // ── Disposal ──────────────────────────────────────────────────────────

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

/// <summary>
/// Intermediate activations for one transformer block.
/// Needed to compute the backward pass through attention, FFN, and norms.
/// </summary>
public sealed class BlockContext : IDisposable
{
    private bool _disposed;

    // ── Norm 1 (pre-attention) ────────────────────────────────────────────

    /// <summary>Input to norm1 — the residual stream before this block.</summary>
    public Tensor<float>? Norm1Input { get; set; }

    /// <summary>Norm1 output = input to attention. Also stores rmsInv per row.</summary>
    public Tensor<float>? Norm1Out   { get; set; }

    /// <summary>Per-row rmsInv values used in RMSNorm backward.</summary>
    public float[]? Norm1RmsInv { get; set; }

    // ── Attention ─────────────────────────────────────────────────────────

    public Tensor<float>? Q          { get; set; }
    public Tensor<float>? K          { get; set; }
    public Tensor<float>? V          { get; set; }
    public Tensor<float>? AttnProbs  { get; set; }   // softmax(QK^T/√d) [B,H,S,S]
    public Tensor<float>? AttnOut    { get; set; }   // before output projection
    public Tensor<float>? AttnProjOut { get; set; }  // after output projection

    // ── Norm 2 (pre-FFN) ──────────────────────────────────────────────────

    public Tensor<float>? Norm2Input { get; set; }
    public Tensor<float>? Norm2Out   { get; set; }
    public float[]? Norm2RmsInv { get; set; }

    // ── FFN ───────────────────────────────────────────────────────────────

    public Tensor<float>? FfnHidden  { get; set; }   // after W1 (or gate/up for gated)
    public Tensor<float>? FfnGate    { get; set; }   // gated FFN: gate projection
    public Tensor<float>? FfnUp      { get; set; }   // gated FFN: up projection
    public Tensor<float>? FfnActOut  { get; set; }   // after activation
    public Tensor<float>? FfnOut     { get; set; }   // after W2/Wdown

    // ── Disposal ──────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Norm1Input?.Dispose();  Norm1Out?.Dispose();
        Q?.Dispose();  K?.Dispose();  V?.Dispose();
        AttnProbs?.Dispose();  AttnOut?.Dispose();  AttnProjOut?.Dispose();
        Norm2Input?.Dispose();  Norm2Out?.Dispose();
        FfnHidden?.Dispose();  FfnGate?.Dispose();  FfnUp?.Dispose();
        FfnActOut?.Dispose();  FfnOut?.Dispose();
    }
}
