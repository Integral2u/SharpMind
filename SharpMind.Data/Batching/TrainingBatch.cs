using SharpMind.Core.Tensors;

namespace SharpMind.Data.Batching;
/// <summary>
/// A single training batch ready to be fed into a model.
///
/// <see cref="TokenIds"/>  shape: [BatchSize, SeqLen]  — input token IDs.
/// <see cref="Labels"/>    shape: [BatchSize, SeqLen]  — target token IDs (shifted by 1 for CLM).
/// <see cref="AttentionMask"/> shape: [BatchSize, SeqLen] — 1 for real tokens, 0 for padding.
/// </summary>
public sealed class TrainingBatch(
    Tensor<int> tokenIds,
    Tensor<int> labels,
    Tensor<float> attentionMask,
    int realTokenCount) : IDisposable
{
    public Tensor<int> TokenIds { get; } = tokenIds;
    public Tensor<int> Labels { get; } = labels;
    public Tensor<float> AttentionMask { get; } = attentionMask;

    /// <summary>Number of real (non-padding) tokens across the batch.</summary>
    public int RealTokenCount { get; } = realTokenCount;

    public void Dispose()
    {
        TokenIds.Dispose();
        Labels.Dispose();
        AttentionMask.Dispose();
    }
}
