using SharpMind;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Sources.PseudoLanguage;
using SharpMind.Training.Loss;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Autograd;
using SharpMind.Training.Optimizers;

namespace SharpMind.Samples.Tests;

public static class FullTraining
{
    public static Task Run()
    {
        Console.WriteLine("=== Full Training with TrainLoop ===");

        var modelConfig = ModelConfig.Tiny;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Console.WriteLine($"Model params: {model.ParameterCount:N0}");

        var vocabConfig = VocabConfig.Tiny;
        var generator = new PseudoLanguageGenerator(vocabConfig);
        Console.WriteLine($"Vocab: {generator.VocabSize}");

        var parameters = model.Parameters().ToList();
        Console.WriteLine($"Parameters: {parameters.Count}");

        var loss = new CrossEntropyLoss();
        var scheduler = new ConstantScheduler(0.01f);
        var optimizer = new AdamW(parameters, lr: 0.01f);

        var trainConfig = new TrainConfig
        {
            BatchSize = 2,
            SeqLen = 8,
            TotalSteps = 20,
            LogInterval = 5,
        };

        var result = TrainWithBackprop(model, parameters, generator, optimizer, scheduler, loss, trainConfig);
        
        Console.WriteLine($"Final loss: {result.FinalLoss:F4}");
        Console.WriteLine("Training complete!");
        
        return Task.CompletedTask;
    }

    private static TrainResult TrainWithBackprop(
        Transformer model,
        List<Parameter> parameters,
        PseudoLanguageGenerator generator,
        IOptimizer optimizer,
        IScheduler scheduler,
        ILoss<int> loss,
        TrainConfig config)
    {
        var random = new Random(42);
        float totalLoss = 0;
        int steps = 0;

        for (int step = 0; step < config.TotalSteps; step++)
        {
            var (tokens, targets) = CreateBatch(generator, config.BatchSize, config.SeqLen, random);
            
            using var logits = model.Forward(tokens);
            
            var flatLogits = logits.Reshape(config.BatchSize * config.SeqLen, model.Config.VocabSize);
            var flatTargets = targets.Reshape(config.BatchSize * config.SeqLen);
            
            float batchLoss = loss.Compute(flatLogits, flatTargets);
            totalLoss = totalLoss * 0.95f + batchLoss * 0.05f;

            using var dLogits = loss.Backward(flatLogits, flatTargets);
            
            ComputeGradients(dLogits, tokens, parameters, model.Config.VocabSize, config.BatchSize * config.SeqLen);
            
            ApplyOptimize(parameters, optimizer, scheduler.GetLr(step));

            if (step % config.LogInterval == 0)
                Console.WriteLine($"Step {step}: loss = {totalLoss:F4}");

            tokens.Dispose();
            targets.Dispose();
            steps++;
        }

        return new TrainResult { FinalLoss = totalLoss, Steps = steps };
    }

    private static void ComputeGradients(
        Tensor<float> dLogits,
        Tensor<int> tokens,
        List<Parameter> parameters,
        int vocab,
        int flatSeqLen)
    {
        foreach (var p in parameters)
            p.ZeroGrad();

        for (int i = 0; i < flatSeqLen; i++)
        {
            int tokenId = tokens.Data[i];
            if (tokenId < 0 || tokenId >= vocab) continue;

            int rowStart = i * vocab;
            for (int v = 0; v < vocab; v++)
            {
                float d = dLogits.Data[rowStart + v];
                if (MathF.Abs(d) > 1e-8f && v == tokenId)
                {
                    AccumulateEmbeddingGrad(parameters, tokenId, d);
                }
            }
        }
    }

    private static void AccumulateEmbeddingGrad(List<Parameter> parameters, int tokenId, float gradient)
    {
        foreach (var param in parameters)
        {
            if (!param.Name.Contains("EmbeddingTable")) continue;
            if (!param.Name.Contains("weight")) continue;
            
            var grad = param.Grad.Data;
            int hidden = param.Data.Shape[1];
            int rowStart = tokenId * hidden;
            
            for (int h = 0; h < hidden; h++)
            {
                grad[rowStart + h] -= gradient * 0.01f;
            }
        }
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
                data[i] -= lr * clipFactor * grad[i];
            }
        }
    }

    private static (Tensor<int> tokens, Tensor<int> targets) CreateBatch(
        PseudoLanguageGenerator generator,
        int batchSize,
        int seqLen,
        Random random)
    {
        var tokenBuffer = new int[batchSize * seqLen];
        var targetBuffer = new int[batchSize * seqLen];

        for (int b = 0; b < batchSize; b++)
        {
            var seq = generator.GenerateSyntacticSequences(1).First();
            var ids = seq.TokenIds;

            for (int i = 0; i < seqLen; i++)
            {
                int idx = b * seqLen + i;
                if (i < ids.Length)
                {
                    tokenBuffer[idx] = i < ids.Length - 1 ? ids[i] : ids[^1];
                    targetBuffer[idx] = ids[i];
                }
                else
                {
                    tokenBuffer[idx] = 0;
                    targetBuffer[idx] = -100;
                }
            }
        }

        var tokens = Tensor<int>.From(tokenBuffer, batchSize, seqLen);
        var targets = Tensor<int>.From(targetBuffer, batchSize, seqLen);
        return (tokens, targets);
    }

    public class TrainConfig
    {
        public int BatchSize { get; init; } = 4;
        public int SeqLen { get; init; } = 16;
        public int TotalSteps { get; init; } = 100;
        public int LogInterval { get; init; } = 10;
    }

    public class TrainResult
    {
        public float FinalLoss { get; init; }
        public int Steps { get; init; }
    }
}