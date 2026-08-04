using System.Runtime.CompilerServices;
using SharpMind.Core.Tensors;
using SharpMind.Data.Batching;

namespace SharpMind.Data.Sources.PseudoLanguage;

/// <summary>
/// Adapters that expose a <see cref="LearnableGenerator"/> as the same
/// <see cref="TrainingBatch"/> stream consumed by the training loop, so
/// pseudo-language and real (DataLoader-based) data share one code path.
///
/// <c>LearnableGenerator</c> produces raw token-id arrays rather than text, so
/// it cannot flow through the text-tokenising <see cref="DataLoader"/> pipeline;
/// this adapter emits batches directly.
/// </summary>
public static class LearnableGeneratorExtensions
{
    /// <summary>
    /// Streams an unbounded sequence of training batches generated from the
    /// generator's vocabulary. Each batch is [batchSize, seqLen]: row <c>b</c>
    /// holds one freshly generated sample (truncated or zero-padded to
    /// <paramref name="seqLen"/>), labels are the next-token shift with -100 at
    /// each row's last real position and in padding.
    /// </summary>
    /// <param name="generator">The source of synthetic samples.</param>
    /// <param name="batchSize">Rows per batch.</param>
    /// <param name="seqLen">Tokens per row. Samples longer than this are truncated.</param>
    public static async IAsyncEnumerable<TrainingBatch> ToTrainingBatches(
        this LearnableGenerator generator,
        int batchSize,
        int seqLen,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(generator);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(seqLen);

        const int ignoreLabelId = -100;
        const int padTokenId = 0;

        while (!cancellationToken.IsCancellationRequested)
        {
            var tokenIds = new int[batchSize * seqLen];
            var labels = new int[batchSize * seqLen];
            var mask = new float[batchSize * seqLen];
            int realTokenCount = 0;

            for (int b = 0; b < batchSize; b++)
            {
                int[] ids = generator.GenerateTrainingSample().TokenIds;
                int take = Math.Min(ids.Length, seqLen);
                for (int s = 0; s < take; s++)
                {
                    tokenIds[b * seqLen + s] = ids[s];
                    labels[b * seqLen + s] = s + 1 < take ? ids[s + 1] : ignoreLabelId;
                    mask[b * seqLen + s] = 1f;
                    realTokenCount++;
                }
                for (int s = take; s < seqLen; s++)
                {
                    tokenIds[b * seqLen + s] = padTokenId;
                    labels[b * seqLen + s] = ignoreLabelId;
                    mask[b * seqLen + s] = 0f;
                }
            }

            yield return new TrainingBatch(
                Tensor<int>.From(tokenIds, batchSize, seqLen),
                Tensor<int>.From(labels, batchSize, seqLen),
                Tensor<float>.From(mask, batchSize, seqLen),
                realTokenCount);

            await Task.Yield();
        }
    }
}
