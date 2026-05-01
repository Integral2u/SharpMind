using SharpMind.Core.Training.Kernels;
using JigSawDotNet;
using SharpMind.Training.Kernels;

namespace SharpMind.Core.Training;

/// <summary>
/// Manages the mapping of gradient operations to their specific PuzzlePiece implementations.
/// This allows swapping between CPU reference kernels and high-performance GPU kernels.
/// </summary>
public sealed class GradientMapping
{
    public ILinearBackward      Linear      { get; internal set; } = new ValLinearBackward();
    public IRMSNormBackward     RMSNorm     { get; internal set; } = new ValRMSNormBackward();
    public ILayerNormBackward   LayerNorm   { get; internal set; } = new ValLayerNormBackward();
    public IAttentionBackward   Attention   { get; internal set; } = new ValAttentionBackward();
    public IEmbeddingBackward   Embedding   { get; internal set; } = new ValEmbeddingBackward();
    public IActivationBackward  Activation  { get; internal set; } = new ValActivationBackward();

    /// <summary>
    /// Overrides specific kernels with custom implementations.
    /// </summary>
    public void OverrideLinear(ILinearBackward kernel) => Linear = kernel;
    public void OverrideRMSNorm(IRMSNormBackward kernel) => RMSNorm = kernel;
    public void OverrideLayerNorm(ILayerNormBackward kernel) => LayerNorm = kernel;
    public void OverrideAttention(IAttentionBackward kernel) => Attention = kernel;
    public void OverrideEmbedding(IEmbeddingBackward kernel) => Embedding = kernel;
    public void OverrideActivation(IActivationBackward kernel) => Activation = kernel;
}
