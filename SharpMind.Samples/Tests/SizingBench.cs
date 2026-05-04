using SharpMind;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Training.Loss;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Optimizers;
using System.Text;

namespace SharpMind.Samples.Tests;

public static class SizingBench
{
    public static async Task Run()
    {
        Console.WriteLine("=== Model Sizing Benchmark ===");
        Console.WriteLine("Testing combinations of Vocab, Hidden, and Layers...");
        Console.WriteLine("Vocab | Hidden | Layers | Final Loss");
        Console.WriteLine("------------------------------------");

        int[] vocabs = { 16, 64, 128 };
        int[] hiddens = { 16, 32, 64, 128 };
        int[] layers = { 1, 2, 3 };

        var results = new StringBuilder();
        results.AppendLine("Vocab,Hidden,Layers,Loss");

        foreach (var v in vocabs)
        {
            foreach (var h in hiddens)
            {
                foreach (var l in layers)
                {
                    float loss = TrainConfig(v, h, l);
                    Console.WriteLine($"{v,5} | {h,6} | {l,6} | {loss,10:F4}");
                    results.AppendLine($"{v},{h},{l},{loss:F4}");
                }
            }
        }

        await File.WriteAllTextAsync("sizing_results.csv", results.ToString());
        Console.WriteLine("\nResults saved to sizing_results.csv");
    }

    private static float TrainConfig(int v, int hidden, int layers)
    {
        var learnConfig = new LearnableConfig
        {
            BatchSize = 4,
            SeqLen = 4,
            TrainSamples = 200,
            // Minimum required for NounVerbNoun
            IncludeNouns = true,
            IncludeVerbs = true,
            IncludeObjects = true,
            // Additional for larger vocabs
            IncludeAdjectives = v >= 128,
            IncludeAdverbs = v >= 128,
            IncludeQuestions = v >= 128,
            IncludePronouns = v >= 128,
            SyntaxPattern = v >= 128 ? SyntaxPattern.QuerySubjectEat : SyntaxPattern.NounVerbNoun,
        };

        var generator = new LearnableGenerator(learnConfig);
        
        var modelConfig = new ModelConfig
        {
            VocabSize = generator.VocabSize,
            HiddenDim = hidden,
            NumLayers = layers,
            NumHeads = 4,
            NumKvHeads = 4,
            FfnDim = hidden * 4,
            MaxSeqLen = 16,
        };
        modelConfig.Validate();

        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        
        var parameters = model.Parameters().ToList();
        var lossFn = new CrossEntropyLoss();
        var scheduler = new ConstantScheduler(0.01f);
        var optimizer = new AdamW(parameters, lr: 0.01f);

        float totalLoss = 0;
        int totalSteps = 100;

        for (int step = 0; step < totalSteps; step++)
        {
            var batch = generator.GenerateBatch(learnConfig.BatchSize);
            int maxSeqLen = batch.TokenIds.Length / learnConfig.BatchSize;
            
            var tokens = Tensor<int>.From(batch.TokenIds, learnConfig.BatchSize, maxSeqLen);
            var targets = CreateTargets(batch, learnConfig.BatchSize, maxSeqLen);
            
            using var logits = model.Forward(tokens);
            var flatLogits = logits.Reshape(learnConfig.BatchSize * maxSeqLen, model.Config.VocabSize);
            var flatTargets = targets.Reshape(learnConfig.BatchSize * maxSeqLen);
            
            float batchLoss = lossFn.Compute(flatLogits, flatTargets);
            totalLoss = totalLoss * 0.95f + batchLoss * 0.05f;
            
            ApplyOptimize(parameters, optimizer, scheduler.GetLr(step));
            
            tokens.Dispose();
            targets.Dispose();
        }

        return totalLoss;
    }

    private static Tensor<int> CreateTargets(GenerateResult batch, int batchSize, int seqLen)
    {
        int[] targets = new int[batchSize * seqLen];
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen - 1; s++)
            {
                targets[b * seqLen + s] = batch.TokenIds[b * seqLen + s + 1];
            }
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
            int count = param.Grad.Shape.ElementCount;
            for (int i = 0; i < count; i++)
                data[i] = data[i] - lr * grad[i];
        }
    }
}
