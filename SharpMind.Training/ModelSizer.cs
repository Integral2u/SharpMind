using SharpMind;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Sources;
using SharpMind.Tokenizer;
using SharpMind.Tokenizer.Bpe;
using SharpMind.Training.Loss;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Optimizers;
using System.Text;

namespace SharpMind.Training.Sizing;

public record SizingConstraints(
    int MinHiddenDim = 16,
    int MaxHiddenDim = 256,
    int MinLayers = 1,
    int MaxLayers = 8,
    int HiddenStep = 16,
    int LayerStep = 1);

public record SizingBudget(
    int MaxTotalParameters = 10_000_000,
    int SampleSize = 1000,
    int StepsPerConfig = 50);

public static class ModelSizer
{
    public static async Task<ModelConfig> DetermineOptimalConfigAsync(
        IDataSource source, 
        SizingConstraints constraints = null, 
        SizingBudget budget = null)
    {
        constraints ??= new SizingConstraints();
        budget ??= new SizingBudget();

        Console.WriteLine($"--- Starting Auto-Sizing on source: {source.Description} ---");

        // 1. Sample data and train a temporary tokenizer
        var sampleTexts = new List<string>();
        await foreach (var text in source.ReadAsync())
        {
            sampleTexts.Add(text);
            if (sampleTexts.Count >= budget.SampleSize) break;
        }

        Console.WriteLine($"Sampled {sampleTexts.Count} documents. Training BPE...");
        var trainer = new BpeTrainer(targetVocabSize: 1024);
        var bpeModel = await trainer.TrainAsync(SampleToAsyncEnumerable(sampleTexts));
        var tokenizer = new SharpMind.Tokenizer.Tokenizer(bpeModel);
        int vocabSize = tokenizer.VocabSize;

        // 2. Grid search over hyperparameters
        var bestConfig = (ModelConfig)null;
        float bestEfficiency = float.NegativeInfinity;

        for (int h = constraints.MinHiddenDim; h <= constraints.MaxHiddenDim; h += constraints.HiddenStep)
        {
            for (int l = constraints.MinLayers; l <= constraints.MaxLayers; l += constraints.LayerStep)
            {
                var config = new ModelConfig
                {
                    VocabSize = vocabSize,
                    HiddenDim = h,
                    NumLayers = l,
                    NumHeads = 4, // Fixed for sizing
                    NumKvHeads = 4,
                    FfnDim = h * 4,
                    MaxSeqLen = 128,
                };

                if (h % 4 != 0) continue; // Ensure divisible by NumHeads

                long paramCount = CalculateParamCount(config);
                if (paramCount > budget.MaxTotalParameters) continue;

                float loss = await EvaluateConfigAsync(config, tokenizer, sampleTexts, budget.StepsPerConfig);
                
                // Efficiency = (Loss Reduction / Parameter Count)
                // Since we don't have a baseline, we use a simple score: -Loss / log(Params)
                // A better way is to compare against the smallest model.
                float efficiency = -loss / MathF.Log10(paramCount + 1);

                Console.WriteLine($"Testing: H={h}, L={l}, Params={paramCount:N0}, Loss={loss:F4}, Eff={efficiency:F4}");

                if (efficiency > bestEfficiency)
                {
                    bestEfficiency = efficiency;
                    bestConfig = config;
                }
            }
        }

        Console.WriteLine($"--- Sizing Complete. Optimal Config: Hidden={bestConfig?.HiddenDim}, Layers={bestConfig?.NumLayers} ---");
        return bestConfig;
    }

    private static async Task<float> EvaluateConfigAsync(ModelConfig config, SharpMind.Tokenizer.Tokenizer tokenizer, List<string> samples, int steps)
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var model = ModelFactory.Create(config, sharpConfig);
        var parameters = model.Parameters().ToList();
        var lossFn = new CrossEntropyLoss();
        var scheduler = new ConstantScheduler(0.01f);
        var optimizer = new AdamW(parameters, lr: 0.01f);

        float totalLoss = 0;
        int batchSize = 4;
        int seqLen = 64;

        for (int step = 0; step < steps; step++)
        {
            // Random batch from samples
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

            var tokensTensor = Tensor<int>.From(flatTokens, batchSize, seqLen);
            var targetsTensor = CreateTargets(flatTokens, batchSize, seqLen);

            using var logits = model.Forward(tokensTensor);
            var flatLogits = logits.Reshape(batchSize * seqLen, config.VocabSize);
            var flatTargets = targetsTensor.Reshape(batchSize * seqLen);

            float batchLoss = lossFn.Compute(flatLogits, flatTargets);
            totalLoss = totalLoss * 0.9f + batchLoss * 0.1f;

            ApplyOptimize(parameters, optimizer, scheduler.GetLr(step));

            tokensTensor.Dispose();
            targetsTensor.Dispose();
        }

        return totalLoss;
    }

    private static long CalculateParamCount(ModelConfig config)
    {
        // Very rough approximation of params
        long embedding = (long)config.VocabSize * config.HiddenDim;
        long block = (long)config.NumLayers * (config.HiddenDim * config.HiddenDim * 12); // Attn + FFN
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

    private static void ApplyOptimize(List<Parameter> parameters, IOptimizer optimizer, float lr)
    {
        optimizer.LearningRate = lr;
        foreach (var param in parameters)
        {
            var data = param.Data.Data;
            var grad = param.Grad.Data;
            for (int i = 0; i < data.Length; i++)
                data[i] -= lr * grad[i];
        }
    }

    private static async IAsyncEnumerable<string> SampleToAsyncEnumerable(List<string> samples)
    {
        foreach (var s in samples) yield return s;
        await Task.CompletedTask;
    }
}
