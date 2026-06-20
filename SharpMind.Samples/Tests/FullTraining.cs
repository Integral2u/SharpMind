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
        
        var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig);
        using var model = ModelFactory.CreateSession(weights, sharpConfig);
        Console.WriteLine($"Model params: {model.ParameterCount:N0}");

        var learnConfig = new LearnableConfig
        {
            BatchSize = 4,
            SeqLen = 4,
            TrainSamples = 200,
            TestSamples = 50,
            IncludeNouns = true,
            IncludeVerbs = true,
            IncludeObjects = true,
            IncludeAdjectives = false,
            IncludeQuestions = true,
            IncludePronouns = true,
            SyntaxPattern = SyntaxPattern.QuerySubjectEat,
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
            int maxSeqLen = batch.TokenIds.Length / config.BatchSize;
            
            var tokens = Tensor<int>.From(batch.TokenIds, config.BatchSize, maxSeqLen);
            var targets = CreateTargets(batch, config.BatchSize, maxSeqLen);
            
            using var logits = model.Forward(tokens);
            
            var flatLogits = logits.Reshape(config.BatchSize * maxSeqLen, model.Config.VocabSize);
            var flatTargets = targets.Reshape(config.BatchSize * maxSeqLen);
            
            float batchLoss = loss.Compute(flatLogits, flatTargets);
            totalLoss = totalLoss * 0.95f + batchLoss * 0.05f;

            using var dLogits = loss.Backward(flatLogits, flatTargets);
            
            ComputeGradientsLearnable(tokens, parameters, model.Config.VocabSize, config.BatchSize, maxSeqLen, model);
            
            ApplyOptimize(parameters, optimizer, scheduler.GetLr(step));

            if (step % 20 == 0)
            {
                var acc = ComputeAccuracy(logits, batch);
                Console.WriteLine($"Step {step}: loss = {totalLoss:F4}, next-token acc = {acc * 100:F1}%");
            }
            
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

    private static void ComputeGradientsLearnable(
        Tensor<int> tokens,
        List<Parameter> parameters,
        int vocab,
        int batch,
        int seqLen,
        Transformer model)
    {
        foreach (var p in parameters)
            p.ZeroGrad();

        var param = parameters.FirstOrDefault(p => p.Name.Contains("EmbeddingTable") && p.Name.Contains("weight"));
        if (param == null) return;
        
        var data = param.Data.Data;
        var grad = param.Grad.Data;
        int count = param.Data.Shape.ElementCount;
        
        int sampleCount = Math.Min(count, 500);
        
        for (int i = 0; i < sampleCount; i++)
        {
            float original = data[i];
            
            data[i] = original + 1e-3f;
            using var logitsPlus = model.Forward(tokens);
            var flatPlus = logitsPlus.Reshape(batch * seqLen, vocab);
            var targetsPlus = Tensor<int>.Zeros(batch * seqLen);
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seqLen - 1; s++)
                    targetsPlus.Data[b * seqLen + s] = tokens.Data[b * seqLen + s + 1];
            var lossPlus = ComputeLoss(flatPlus, targetsPlus);
            flatPlus.Dispose();
            targetsPlus.Dispose();
            logitsPlus.Dispose();
            
            data[i] = original - 1e-3f;
            using var logitsMinus = model.Forward(tokens);
            var flatMinus = logitsMinus.Reshape(batch * seqLen, vocab);
            var targetsMinus = Tensor<int>.Zeros(batch * seqLen);
            for (int b = 0; b < batch; b++)
                for (int s = 0; s < seqLen - 1; s++)
                    targetsMinus.Data[b * seqLen + s] = tokens.Data[b * seqLen + s + 1];
            var lossMinus = ComputeLoss(flatMinus, targetsMinus);
            flatMinus.Dispose();
            targetsMinus.Dispose();
            logitsMinus.Dispose();
            
            data[i] = original;
            
            grad[i] = (lossPlus - lossMinus) / (2e-3f);
        }
        
        float gradNorm = 0f;
        for (int i = 0; i < sampleCount; i++)
            gradNorm += grad[i] * grad[i];
        gradNorm = MathF.Sqrt(gradNorm / sampleCount + 1e-8f);
        
        if (gradNorm > 1f)
        {
            float scale = 1f / gradNorm;
            for (int i = 0; i < sampleCount; i++)
                grad[i] *= scale;
        }
    }
    
    private static float ComputeLoss(Tensor<float> logits, Tensor<int> targets)
    {
        int total = 0;
        float loss = 0;
        int vocab = logits.Shape.Cols;
        
        for (int i = 0; i < targets.Shape.ElementCount; i++)
        {
            int t = targets.Data[i];
            if (t < 0) continue;
            
            float maxLogit = float.NegativeInfinity;
            for (int v = 0; v < vocab; v++)
                if (logits.Data[i * vocab + v] > maxLogit)
                    maxLogit = logits.Data[i * vocab + v];
            
            float sum = 0;
            for (int v = 0; v < vocab; v++)
                sum += MathF.Exp(logits.Data[i * vocab + v] - maxLogit);
            
            float predLogProb = logits.Data[i * vocab + t] - maxLogit - MathF.Log(sum);
            loss -= predLogProb;
            total++;
        }
        
        return total > 0 ? loss / total : 0f;
    }
    private static float ComputeAccuracy(Tensor<float> logits, GenerateResult batch)
    {
        int batchSize = batch.BatchSize;
        int vocab = logits.Shape[2];
        int seqLen = logits.Shape[1];
        
        int correct = 0;
        int total = 0;
        
        for (int b = 0; b < batchSize; b++)
        {
            for (int s = 0; s < seqLen - 1; s++)
            {
                int trueToken = batch.TokenIds[b * seqLen + s + 1];
                if (trueToken < 0 || trueToken >= vocab) continue;
                
                float maxLogit = float.NegativeInfinity;
                int predToken = -1;
                
                int idx = (b * seqLen + s) * vocab;
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
                total++;
            }
        }
        
        return total > 0 ? (float)correct / total : 0f;
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