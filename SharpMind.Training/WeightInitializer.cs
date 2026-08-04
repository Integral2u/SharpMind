using SharpMind.Model;

namespace SharpMind.Training;

/// <summary>
/// Initialises freshly allocated training weights.
/// <see cref="ModelFactory.CreateForTraining"/> allocates zeroed tensors which
/// would otherwise pin the loss at log(V); this randomises the embedding and
/// attention/FFN matrices so finite-difference and backprop gradients are
/// meaningful from step 0.
/// </summary>
public static class WeightInitializer
{
    /// <summary>
    /// Randomises all float weights (GPT-2 style, std 0.02) in place.
    /// Norm weights are left untouched — they are allocated as Ones by
    /// <see cref="ModelFactory.CreateForTraining"/> and must not be re-randomised.
    /// </summary>
    /// <param name="weights">The training weight set to populate.</param>
    /// <param name="seed">Random seed for reproducible initialisation.</param>
    /// <param name="std">Standard deviation of the Gaussian draws.</param>
    /// <param name="progress">Optional 0..1 progress reporter (per tensor).</param>
    public static void InitializeRandomly(
        TransformerWeights weights,
        int seed,
        float std = 0.02f,
        IProgress<float>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(weights);

        var rng = new Random(seed);

        int total = 1 + weights.Blocks.Length * 6; // embedding + 6 matrices per block
        int done = 0;

        FillNormal(weights.EmbeddingWeight.Data, rng, std);
        progress?.Report((float)++done / total);

        foreach (var block in weights.Blocks)
        {
            FillNormal(block.Wq!.Data, rng, std);
            FillNormal(block.Wk!.Data, rng, std);
            FillNormal(block.Wv!.Data, rng, std);
            FillNormal(block.Wo!.Data, rng, std);
            FillNormal(block.Wf1!.Data, rng, std);
            FillNormal(block.Wf2!.Data, rng, std);
            progress?.Report((float)++done / total);
        }
    }

    /// <summary>Fills <paramref name="data"/> with independent standard-normal draws scaled by <paramref name="std"/>.</summary>
    public static void FillNormal(Span<float> data, Random rng, float std)
    {
        for (int i = 0; i < data.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            data[i] = (float)(std * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
    }
}
