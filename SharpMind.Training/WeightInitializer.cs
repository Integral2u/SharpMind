using SharpMind.Core.Tensors;
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
        if (weights.PositionEmbedding is not null) total++;
        int done = 0;

        FillNormal(weights.EmbeddingWeight.Data, rng, std);
        progress?.Report((float)++done / total);

        if (weights.PositionEmbedding is { } posEmb)
        {
            FillNormal(posEmb.Data, rng, std);
            progress?.Report((float)++done / total);
        }

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

    /// <summary>
    /// Randomises MoE router and per-expert weights (GPT-2 style, std 0.02) in
    /// place. Unlike <see cref="InitializeRandomly"/> this runs on the assembled
    /// <see cref="Transformer"/>, because router/expert tensors live on the
    /// built <see cref="Model.Layers.Ffn.FfnLayer"/> rather than in
    /// <see cref="TransformerWeights.BlockWeights"/>.
    /// </summary>
    /// <param name="model">The assembled training transformer to populate.</param>
    /// <param name="seed">Random seed for reproducible initialisation.</param>
    /// <param name="std">Standard deviation of the Gaussian draws.</param>
    public static void InitializeModelMoE(Transformer model, int seed, float std = 0.02f)
    {
        ArgumentNullException.ThrowIfNull(model);

        var rng = new Random(seed);
        for (int i = 0; i < model.Config.NumLayers; i++)
        {
            var ffn = model.GetBlock(i)?.Ffn;
            var router = ffn?.RouterLayer;
            if (router is null) continue;

            FillNormal(router.Weight.Data, rng, std);
            if (router.Bias is { } rb) FillNormal(rb.Data, rng, std);

            if (ffn!.ExpertGateLayers is { } gateLayers)
                foreach (var l in gateLayers) FillLayer(l, rng, std);
            if (ffn.ExpertUpLayers is { } upLayers)
                foreach (var l in upLayers) FillLayer(l, rng, std);
            if (ffn.ExpertDownLayers is { } downLayers)
                foreach (var l in downLayers) FillLayer(l, rng, std);
        }
    }

    private static void FillLayer(Model.Layers.LinearLayer layer, Random rng, float std)
    {
        FillNormal(layer.Weight.Data, rng, std);
        if (layer.Bias is { } b) FillNormal(b.Data, rng, std);
    }

    /// <summary>
    /// Initialises the token rows appended to a grown embedding table. The first
    /// <paramref name="existingRows"/> rows are left untouched (they carry the
    /// reused weights); rows beyond are filled with Gaussian draws scaled to the
    /// existing rows' standard deviation, so new tokens start in the same region
    /// of embedding space as the ones the model already learned.
    /// </summary>
    /// <param name="embedding">The [VocabSize, HiddenDim] embedding tensor.</param>
    /// <param name="existingRows">How many leading rows hold reused weights.</param>
    /// <param name="seed">Random seed for reproducible initialisation.</param>
    public static void InitializeNewEmbeddingRows(Tensor<float> embedding, int existingRows, int seed)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        if (embedding.Rank != 2) return;

        int rows = embedding.Shape.Rows;
        int cols = embedding.Shape.Cols;
        if (existingRows >= rows) return;
        if (existingRows < 0) existingRows = 0;

        // Estimate the scale of the reused rows (std of a sampled subset, to keep
        // the cost bounded on large tables), then match it for the new rows.
        float std = ExistingStd(embedding, existingRows, cols);

        var rng = new Random(seed);
        var data = embedding.Data;
        for (int r = existingRows; r < rows; r++)
            FillNormal(data.Slice(r * cols, cols), rng, std);
    }

    private static float ExistingStd(Tensor<float> embedding, int existingRows, int cols)
    {
        if (existingRows <= 0) return 0.02f;

        // Mean and variance over at most ~4096 rows to bound the work.
        int sampleRows = Math.Min(existingRows, 4096);
        int stride = Math.Max(1, existingRows / sampleRows);
        var data = embedding.Data;

        double sum = 0, sumSq = 0;
        long n = 0;
        for (int r = 0; r < existingRows; r += stride)
        {
            var row = data.Slice(r * cols, cols);
            for (int c = 0; c < cols; c++)
            {
                double v = row[c];
                sum += v;
                sumSq += v * v;
                n++;
            }
        }
        if (n == 0) return 0.02f;
        double mean = sum / n;
        double var = Math.Max(0, sumSq / n - mean * mean);
        double std = Math.Sqrt(var);
        // Guard against a degenerate (all-zero) embedding pinning new rows at 0.
        return std > 1e-9 ? (float)std : 0.02f;
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
