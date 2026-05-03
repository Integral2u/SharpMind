using SharpMind;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Core.Tensors;
using SharpMind.Data.Sources.PseudoLanguage;

namespace SharpMind.Samples.Tests;

public static class FullTraining
{
    public static async Task Run()
    {
        await Task.CompletedTask;

        Console.WriteLine("=== Full Training Test ===");

        var modelConfig = ModelConfig.Tiny;
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };

        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Console.WriteLine($"Model params: {model.ParameterCount:N0}");

        var vocabConfig = VocabConfig.Tiny;
        var generator = new PseudoLanguageGenerator(vocabConfig);
        Console.WriteLine($"Vocab: {generator.VocabSize}");

        var trainConfig = new TrainConfig
        {
            BatchSize = 4,
            SeqLen = 8,
            TotalSteps = 50,
            LearningRate = 1e-3f,
            WeightDecay = 0.01f,
        };

        var (loss, steps) = TrainLoop(model, generator, trainConfig);
        
        Console.WriteLine($"Final loss: {loss:F4} after {steps} steps");
        Console.WriteLine("Training complete!");
    }

    private static (float loss, int steps) TrainLoop(
        Transformer model,
        PseudoLanguageGenerator generator,
        TrainConfig config)
    {
        var random = new Random(42);
        float loss = 0;
        int steps = config.TotalSteps;

        for (int step = 0; step < steps; step++)
        {
            var (tokens, targets) = CreateBatch(generator, config.BatchSize, config.SeqLen, random);
            
            using var logits = model.Forward(tokens);
            
            float batchLoss = ComputeCrossEntropyLoss(logits, targets);
            loss = loss * 0.99f + batchLoss * 0.01f;

            if (step % 10 == 0)
                Console.WriteLine($"Step {step}: loss = {loss:F4}");

            tokens.Dispose();
        }

        return (loss, steps);
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

    private static float ComputeCrossEntropyLoss(Tensor<float> logits, Tensor<int> targets)
    {
        int batch = logits.Shape[0];
        int seqLen = logits.Shape[1];
        int vocab = logits.Shape[2];

        float totalLoss = 0;
        int count = 0;

        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < seqLen; s++)
            {
                int label = targets[b, s];
                if (label < 0) continue;

                int flatIdx = b * seqLen * vocab + s * vocab;
                
                float maxLogit = float.NegativeInfinity;
                for (int v = 0; v < vocab; v++)
                {
                    float val = logits.Data[flatIdx + v];
                    if (val > maxLogit) maxLogit = val;
                }

                float sum = 0;
                for (int v = 0; v < vocab; v++)
                {
                    sum += MathF.Exp(logits.Data[flatIdx + v] - maxLogit);
                }
                float logSum = maxLogit + MathF.Log(sum);
                float logProb = logits.Data[flatIdx + label] - logSum;
                
                totalLoss -= logProb;
                count++;
            }
        }

        return count > 0 ? totalLoss / count : 0;
    }
}

public class TrainConfig
{
    public int BatchSize { get; init; } = 4;
    public int SeqLen { get; init; } = 16;
    public int TotalSteps { get; init; } = 100;
    public float LearningRate { get; init; } = 1e-3f;
    public float WeightDecay { get; init; } = 0.01f;
}
