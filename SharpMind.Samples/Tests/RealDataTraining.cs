using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Sources;
using SharpMind.Data.Parquet.Sources;
using SharpMind.Training.Loss;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Optimizers;

namespace SharpMind.Samples.Tests;

public static class RealDataTraining
{
    public static async Task RunParquet(int size)
    {
        Console.WriteLine("=== Training with Parquet Data ===");
        await Train(new ParquetSource(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets\open-perfectblend\train-*.parquet", "conversations"), size);
    }

    public static async Task RunFusechat(int size)
    {
        Console.WriteLine("=== Training with Fusechat Data ===");
        await Train(new FusechatSource(@"C:\Integral2u\source\repos\SharpMind\ExternalAssets\fusechat_v1\*.json"), size);
    }

    private static async Task Train(IDataSource source, int size)
    {
        Console.WriteLine("Training BPE tokenizer on a subset of data...");
        var trainer = new SharpMind.Tokenizer.Bpe.BpeTrainer(targetVocabSize: size);
        var model = await trainer.TrainAsync(source.ReadAsync());
        var tokenizer = new Tokenizer.Tokenization(model);
        Console.WriteLine($"Tokenizer trained. Vocab size: {tokenizer.VocabSize}");

        var modelConfig = ModelConfig.Learnable;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var modelInstance = ModelFactory.Create(modelConfig, sharpConfig);
        
        var parameters = modelInstance.Parameters().ToList();
        var loss = new CrossEntropyLoss();
        var scheduler = new ConstantScheduler(0.01f);
        var optimizer = new AdamW(parameters, lr: 0.01f);

        int batchSize = 4;
        int seqLen = 128;
        int steps = 100;

        for (int step = 0; step < steps; step++)
        {
            // Collect a batch of strings from the source
            var batchTexts = new List<string>();
            await foreach (var text in source.ReadAsync())
            {
                batchTexts.Add(text);
                if (batchTexts.Count == batchSize) break;
            }

            if (batchTexts.Count < batchSize) break;

            // Tokenize and pad/truncate
            int[][] tokenIds = new int[batchSize][];
            for (int i = 0; i < batchSize; i++)
            {
                var tokens = tokenizer.Encode(batchTexts[i]);
                tokenIds[i] = EnsureLength(tokens, seqLen);
            }

            // Flatten to array
            int[] flatTokens = new int[batchSize * seqLen];
            for (int i = 0; i < batchSize; i++)
                Array.Copy(tokenIds[i], 0, flatTokens, i * seqLen, seqLen);

            var tokensTensor = Tensor<int>.From(flatTokens, batchSize, seqLen);
            var targetsTensor = CreateTargets(flatTokens, batchSize, seqLen);

            // This when _cachedEmbedding = _embedding.Forward(tokenIds); is calls cause VS to crash
            using var logits = modelInstance.Forward(tokensTensor);
            var flatLogits = logits.Reshape(batchSize * seqLen, modelInstance.Config.VocabSize);
            var flatTargets = targetsTensor.Reshape(batchSize * seqLen);

            float batchLoss = loss.Compute(flatLogits, flatTargets);
            
            // Simple gradient update for the sample
            ApplyOptimize(parameters, optimizer, scheduler.GetLr(step));

            if (step % 10 == 0)
                Console.WriteLine($"Step {step}: loss = {batchLoss:F4}");

            tokensTensor.Dispose();
            targetsTensor.Dispose();
        }
    }            

    private static int[] EnsureLength(int[] tokens, int length)
    {
        var result = new int[length];
        Array.Copy(tokens, 0, result, 0, Math.Min(tokens.Length, length));
        for (int i = tokens.Length; i < length; i++) result[i] = -100; // Padding
        return result;
    }

    private static Tensor<int> CreateTargets(int[] flatTokens, int batchSize, int seqLen)
    {
        int[] targets = new int[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen - 1; s++)
            {
                targets[b * seqLen + s] = flatTokens[b * seqLen + s + 1];
            }
            targets[b * seqLen + seqLen - 1] = -100;
        }
        return Tensor<int>.From(targets, batchSize, seqLen);
    }

    private static void ApplyOptimize(List<Parameter> parameters, AdamW optimizer, float lr)
    {
        optimizer.LearningRate = lr;
        foreach (var param in parameters)
        {
            var data = param.Data.Data;
            var grad = param.Grad.Data;
            int count = param.Grad.Shape.ElementCount;
            for (int i = 0; i < count; i++)
                data[i] = data[i] - lr * grad[i];
        }
    }
}
