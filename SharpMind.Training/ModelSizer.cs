using SharpMind.Core;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Core.Tensors;
using SharpMind.Data.Sources;
using SharpMind.Tokenization;
using SharpMind.Tokenization.Bpe;
using SharpMind.Training.Loss;

namespace SharpMind.Training;

public static class ModelSizer
{
    public static async Task<ModelConfig> DetermineOptimalConfigAsync(
        IDataSource source, 
        SizingConstraints? constraints = null, 
        SizingBudget? budget = null,
        IProgress<float>? progress = null,
        CancellationToken ct = default)
    {
        constraints ??= new SizingConstraints();
        budget ??= new SizingBudget();     

        // 1. Sample data and train a temporary tokenizer
        var sampleTexts = new List<string>();
        await foreach (var text in source.ReadAsync(ct))
        {
            sampleTexts.Add(text);
            if (sampleTexts.Count >= budget.SampleSize) break;
        }

        if (sampleTexts.Count == 0)
            throw new InvalidOperationException("Source yielded no documents. Cannot determine optimal size.");

        var trainer = new BpeTrainer(targetVocabSize: 1024);
        var bpeModel = await trainer.TrainAsync(SampleToAsyncEnumerable(sampleTexts),ct);
        var tokenizer = new Tokenizer(bpeModel);
        int vocabSize = tokenizer.VocabSize;

        // 2. Grid search over hyperparameters
        var configsAndLosses = new List<(ModelConfig Config, float Loss, long Params)>();
        var candidateSizes = new List<(int H, int L)>();
        for (int h = constraints.MinHiddenDim; h <= constraints.MaxHiddenDim; h += constraints.HiddenStep)
            for (int l = constraints.MinLayers; l <= constraints.MaxLayers; l += constraints.LayerStep)
                if (h % 4 == 0)
                    candidateSizes.Add((h, l));

        int totalCandidates = candidateSizes.Count;
        int evaluated = 0;
        progress?.Report(0f);

        foreach (var (h, l) in candidateSizes)
        {
            ct.ThrowIfCancellationRequested();

            var config = new ModelConfig
            {
                VocabSize = vocabSize,
                HiddenDim = h,
                NumLayers = l,
                NumHeads = 4,
                NumKvHeads = 4,
                FfnDim = h * 4,
                MaxSeqLen = 128,
            };

            long paramCount = CalculateParamCount(config);
            if (paramCount > budget.MaxTotalParameters) continue;

            // Use Task.Run to avoid async warnings and run on background thread
            float loss = await Task.Run(() => EvaluateConfig(config, tokenizer, sampleTexts, budget.StepsPerConfig), ct);
            
            configsAndLosses.Add((config, loss, paramCount));
            evaluated++;
            progress?.Report((float)evaluated / totalCandidates);
        }

        // 3. Elbow Point Analysis
        // Sort by parameter count
        var sorted = configsAndLosses.OrderBy(x => x.Params).ToList();
        if (sorted.Count == 0) throw new Exception("No valid configurations were tested.");

        float baselineLoss = sorted[0].Loss;
        float minLoss = sorted.Min(x => x.Loss);
        float totalImprovement = baselineLoss - minLoss;

        ModelConfig bestConfig = sorted[0].Config;
        
        // Find the smallest model that achieves at least 90% of the maximum improvement
        foreach (var (config, loss, paramsCount) in sorted)
        {
            float improvement = baselineLoss - loss;
            if (improvement >= 0.9f * totalImprovement)
            {
                bestConfig = config;
                break;
            }
        }

        return bestConfig;
    }

    private static float EvaluateConfig(ModelConfig config, Tokenizer tokenizer, List<string> samples, int steps)
    {
        // The training forward path is what the eventual trainer will use, and
        // it avoids the inference/JIT linear path entirely (which needs a
        // hardware mapping). Gradients are full-transformer backprop, which is
        // not yet wired in the training loops, so this measures each config's
        // initial forward loss — enough to rank small vs large configs.
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(config, sharpConfig);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);

        var lossFn = new CrossEntropyLoss();

        float totalLoss = 0;
        int batchSize = 4;
        int seqLen = 64;

        for (int step = 0; step < steps; step++)
        {
            var batchStrings = new List<string>();
            for (int i = 0; i < batchSize; i++)
                batchStrings.Add(samples[Random.Shared.Next(samples.Count)]);

            int[][] tokenIds = new int[batchSize][];
            for (int i = 0; i < batchSize; i++)
            {
                var tokens = tokenizer.Encode(batchStrings[i]);
                tokenIds[i] = EnsureLength(tokens, seqLen, tokenizer.PadId);
            }

            int[] flatTokens = new int[batchSize * seqLen];
            for (int i = 0; i < batchSize; i++)
                Array.Copy(tokenIds[i], 0, flatTokens, i * seqLen, seqLen);

            using var tokensTensor = Tensor<int>.From(flatTokens, batchSize, seqLen);
            using var targetsTensor = CreateTargets(flatTokens, batchSize, seqLen);

            using var logits = model.Forward(tokensTensor);
            using var flatLogits = logits.Reshape(batchSize * seqLen, config.VocabSize);
            using var flatTargets = targetsTensor.Reshape(batchSize * seqLen);

            float batchLoss = lossFn.Compute(flatLogits, flatTargets);
            totalLoss = totalLoss * 0.9f + batchLoss * 0.1f;
        }

        return totalLoss;
    }

    private static long CalculateParamCount(ModelConfig config)
    {
        long embedding = (long)config.VocabSize * config.HiddenDim;
        long block = (long)config.NumLayers * (config.HiddenDim * config.HiddenDim * 12); 
        return embedding + block;
    }

    private static int[] EnsureLength(int[] tokens, int length, int padId)
    {
        var result = new int[length];
        Array.Copy(tokens, 0, result, 0, Math.Min(tokens.Length, length));
        for (int i = tokens.Length; i < length; i++) result[i] = padId;
        return result;
    }

    private static Tensor<int> CreateTargets(int[] flatTokens, int batchSize, int seqLen)
    {
        int[] targets = new int[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen - 1; s++)
                targets[b * seqLen + s] = flatTokens[b * seqLen + s + 1];
            targets[b * seqLen + seqLen - 1] = -100;
        }
        return Tensor<int>.From(targets, batchSize, seqLen);
    }

    private static async IAsyncEnumerable<string> SampleToAsyncEnumerable(List<string> samples)
    {
        foreach (var s in samples) yield return s;
        await Task.CompletedTask;
    }
}
