using SharpMind.Model;
using SharpMind.Data.Batching;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Core.Tensors;
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

    /// <summary>
    /// Measures greedy next-token accuracy on <paramref name="samples"/> fresh
    /// pseudo-language sequences. Each position is scored against the token that
    /// actually follows it in the generated sample.
    /// </summary>
    public float NextTokenAccuracy(LearnableGenerator generator, int vocab, int samples = 20, IProgress<float>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(generator);
        var ids = new int[samples][];
        for (int i = 0; i < samples; i++)
        {
            ids[i] = generator.GenerateTrainingSample().TokenIds;
            progress?.Report((float)i / samples);
        }
        return NextTokenAccuracy(ids, vocab);
    }

    /// <summary>
    /// Measures greedy next-token accuracy over the supplied token-ID samples.
    /// Row <c>s</c> is predicted from the tokens <c>[0..s)</c> and scored against
    /// <c>ids[s + 1]</c> (the true next token).
    /// </summary>
    public float NextTokenAccuracy(IEnumerable<int[]> sampleTokenIds, int vocab)
    {
        ArgumentNullException.ThrowIfNull(sampleTokenIds);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(vocab);

        int total = 0;
        int correct = 0;
        foreach (var ids in sampleTokenIds)
        {
            if (ids.Length < 2) continue;
            using var tokens = Tensor<int>.From(ids, 1, ids.Length);
            using var logits = _model.Forward(tokens);
            for (int s = 0; s < ids.Length - 1; s++)
            {
                int target = ids[s + 1];
                float max = float.NegativeInfinity;
                int best = -1;
                for (int v = 0; v < vocab; v++)
                {
                    float l = logits.Data[s * vocab + v];
                    if (float.IsFinite(l) && l > max) { max = l; best = v; }
                }
                total++;
                if (best == target) correct++;
            }
        }
        return total > 0 ? (float)correct / total : 0f;
    }
}
