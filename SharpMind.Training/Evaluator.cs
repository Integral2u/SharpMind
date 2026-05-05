using SharpMind.Model;
using SharpMind.Data.Batching;
using SharpMind.Core.Training;

namespace SharpMind.Training;

/// <summary>
/// Evaluates model performance on a validation dataset.
/// </summary>
public sealed class Evaluator(Transformer model, ILoss<int> lossFn)
{
    private readonly Transformer _model = model;
    private readonly ILoss<int> _lossFn = lossFn;

    /// <summary>
    /// Computes the average cross-entropy loss and perplexity over a set of batches.
    /// </summary>
    public (float Loss, float Perplexity) Evaluate(IEnumerable<TrainingBatch> batches)
    {
        float totalLoss = 0;
        int count = 0;

        foreach (var batch in batches)
        {
            using var logits = _model.Forward(batch.TokenIds);
            
            int batchSize = batch.TokenIds.Shape[0];
            int seqLen = batch.TokenIds.Shape[1];
            using var flatLogits = logits.Reshape(batchSize * seqLen, _model.Config.VocabSize);
            using var flatLabels = batch.Labels.Reshape(batchSize * seqLen);

            totalLoss += _lossFn.Compute(flatLogits, flatLabels);
            count++;
        }

        float avgLoss = count > 0 ? totalLoss / count : 0;
        // Perplexity = exp(CrossEntropyLoss)
        float perplexity = MathF.Exp(avgLoss);

        return (avgLoss, perplexity);
    }
}
