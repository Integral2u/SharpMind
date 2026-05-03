using SharpMind;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Training.Loss;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Optimizers;

namespace SharpMind.Samples.Tests;

public static class FullTraining
{
    public static Task Run()
    {
        Console.WriteLine("=== Full Training with Learnable Pseudo-Language ===");

        var modelConfig = ModelConfig.Learnable;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Console.WriteLine($"Model params: {model.ParameterCount:N0}");

        var learnConfig = new LearnableConfig
        {
            BatchSize = 4,
            SeqLen = 3,
            TrainSamples = 200,
            TestSamples = 50,
            IncludeNouns = true,
            IncludeVerbs = true,
            IncludeObjects = true,
            IncludeAdjectives = false,
            SyntaxPattern = SyntaxPattern.NounVerbNoun,
        };

        var generator = new LearnableGenerator(learnConfig);
        Console.WriteLine($"Learnable vocab: {generator.VocabSize}");
        Console.WriteLine($"Syntax pattern: {learnConfig.SyntaxPattern}");
        Console.WriteLine($"Complexity score: {learnConfig.ComplexityScore}");
        Console.WriteLine($"Words: {string.Join(", ", generator.Vocabulary.Take(16))}...");

        var parameters = model.Parameters().ToList();
        Console.WriteLine($"Parameters: {parameters.Count}");

        var loss = new CrossEntropyLoss();
        var scheduler = new ConstantScheduler(0.01f);
        var optimizer = new AdamW(parameters, lr: 0.01f);

        var finalLoss = TrainLearnable(model, parameters, generator, optimizer, scheduler, loss, learnConfig);
        
        Console.WriteLine($"\n=== Final Results ===");
        Console.WriteLine($"Steps: 200");
        Console.WriteLine($"Final loss: {finalLoss:F4}");

        return Task.CompletedTask;
    }

    private static float TrainLearnable(
        Transformer model,
        List<Parameter> parameters,
        LearnableGenerator generator,
        IOptimizer optimizer,
        IScheduler scheduler,
        ILoss<int> loss,
        LearnableConfig config)
    {
        float totalLoss = 0;
        int totalSteps = 200;

        for (int step = 0; step < totalSteps; step++)
        {
            var batch = generator.GenerateBatch(config.BatchSize);
            var tokens = Tensor<int>.From(batch.TokenIds, config.BatchSize, 3);
            var targets = CreateTargets(batch, config.BatchSize);
            
            using var logits = model.Forward(tokens);
            
            var flatLogits = logits.Reshape(12, model.Config.VocabSize);
            var flatTargets = targets.Reshape(12);
            
            float batchLoss = loss.Compute(flatLogits, flatTargets);
            totalLoss = totalLoss * 0.95f + batchLoss * 0.05f;

            using var dLogits = loss.Backward(flatLogits, flatTargets);
            
            ComputeGradientsLearnable(dLogits, tokens, parameters, model.Config.VocabSize, config.BatchSize, 3, model.Config.HiddenDim);
            
            ApplyOptimize(parameters, optimizer, scheduler.GetLr(step));

            if (step % 20 == 0)
            {
                var acc = ComputeAccuracy(logits, batch);
                Console.WriteLine($"Step {step}: loss = {totalLoss:F4}, acc = {acc * 100:F1}%");
            }
            
            tokens.Dispose();
            targets.Dispose();
        }

        return totalLoss;
    }

    private static Tensor<int> CreateTargets(GenerateResult batch, int batchSize)
    {
        int[] targets = new int[batchSize * 3];
        
        for (int b = 0; b < batchSize; b++)
        {
            int baseIdx = b * 3;
            targets[baseIdx] = batch.TokenIds[baseIdx + 1];
            targets[baseIdx + 1] = batch.TokenIds[baseIdx + 2];
            targets[baseIdx + 2] = -100;
        }
        
        return Tensor<int>.From(targets, batchSize, 3);
    }

    private static void ComputeGradientsLearnable(
        Tensor<float> dLogits,
        Tensor<int> tokens,
        List<Parameter> parameters,
        int vocab,
        int batch,
        int seqLen,
        int hidden)
    {
        foreach (var p in parameters)
            p.ZeroGrad();

        foreach (var param in parameters)
        {
            if (!param.Name.Contains("EmbeddingTable")) continue;
            if (!param.Name.Contains("weight")) continue;
            
            var grad = param.Grad.Data;
            
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int tokenId = tokens.Data[b * seqLen + s];
                    if (tokenId < 0 || tokenId >= vocab) continue;

                    int flatIdx = b * seqLen * vocab + s * vocab;
                    float dL_dLogit = -dLogits.Data[flatIdx + tokenId];
                    
                    int rowStart = tokenId * hidden;
                    for (int h = 0; h < hidden; h++)
                    {
                        grad[rowStart + h] += dL_dLogit;
                    }
                }
            }
        }
    }

    private static float ComputeAccuracy(Tensor<float> logits, GenerateResult batch)
    {
        int batchSize = batch.BatchSize;
        int vocab = logits.Shape[2];
        
        int correct = 0;
        
        for (int b = 0; b < batchSize; b++)
        {
            int trueToken = batch.TokenIds[b * 3 + 1];
            
            float maxLogit = float.NegativeInfinity;
            int predToken = -1;
            
            int idx = b * 3 * vocab + 0 * vocab;
            if (trueToken != -100)
            {
                for (int v = 0; v < vocab; v++)
                {
                    if (logits.Data[idx + v] > maxLogit)
                    {
                        maxLogit = logits.Data[idx + v];
                        predToken = v;
                    }
                }
                
                if (predToken == trueToken)
                    correct++;
            }
        }
        
        return (float)correct / batchSize;
    }

    private static void ApplyOptimize(List<Parameter> parameters, IOptimizer optimizer, float lr)
    {
        optimizer.LearningRate = lr;
        
        foreach (var param in parameters)
        {
            var data = param.Data.Data;
            var grad = param.Grad.Data;
            int count = param.Grad.Shape.ElementCount;
            
            float clipFactor = 1f;
            float gradNorm = 0;
            for (int i = 0; i < count; i++)
                gradNorm += grad[i] * grad[i];
            gradNorm = MathF.Sqrt(gradNorm / count + 1e-8f);
            if (gradNorm > 1f) clipFactor = 1f / gradNorm;

            for (int i = 0; i < count; i++)
            {
                data[i] = data[i] - lr * clipFactor * grad[i];
            }
        }
    }
}