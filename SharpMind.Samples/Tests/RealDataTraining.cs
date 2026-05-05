using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Training.Loss;

namespace SharpMind.Samples.Tests;

public static class RealDataTraining
{
    public static async Task RunPixelChar()
    {
        await Task.CompletedTask;
        Console.WriteLine("=== Training with Pixel Char Data ===");
        
        var source = new PixelCharSource();
        
        var modelConfig = new ModelConfig
        {
            VocabSize = 16,  // Simpler
            HiddenDim = 16,
            NumLayers = 1,
            NumHeads = 1,
            NumKvHeads = 1,
            FfnDim = 32,
            MaxSeqLen = 8,
        };
        modelConfig.Validate();
        
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        
        var rng = new Random(42);
        foreach (var p in model.Parameters())
        {
            var data = p.Data.Data;
            // Use a smaller, safer initialization
            for (int i = 0; i < data.Length; i++)
                data[i] = 0.1f;
        }
        
        // Test very simple forward - just embedding
        Console.WriteLine("Testing simple forward...");
        var testInput = Tensor<int>.From(new int[] { 1, 1, 1, 1, 1, 1, 1, 1 }, 1, 8);
        using var result = model.Forward(testInput);
        
        float[] sample = new float[16];
        for (int i = 0; i < 16; i++) sample[i] = result[i];
        
        // Check for NaN
        bool hasNaN = false;
        foreach (float f in sample)
        {
            if (float.IsNaN(f)) hasNaN = true;
            Console.Write($"{f:F4} ");
        }
        Console.WriteLine();
        
        if (hasNaN)
            Console.WriteLine("NaN detected in output");
        else
            Console.WriteLine("Output looks OK");
        
        int batchSize = 2;
        int steps = 20;

        for (int step = 0; step < steps; step++)
        {
            var batch = source.GetBatch(batchSize);
            
            var tokensTensor = Tensor<int>.From(batch.Inputs, batchSize, 8);
            var targetsTensor = Tensor<int>.From(batch.Targets, batchSize, 8);

            using var logits = model.Forward(tokensTensor);
            int totalTokens = batchSize * 8;
            var flatLogits = logits.Reshape(totalTokens, model.Config.VocabSize);
            var flatTargets = targetsTensor.Reshape(totalTokens);

            var loss = new CrossEntropyLoss();
            float batchLoss = loss.Compute(flatLogits, flatTargets);
            
            // Debug output
            if (step == 0)
            {
                Console.WriteLine($"Logits sample: {flatLogits[0]:F4} {flatLogits[1]:F4} {flatLogits[2]:F4}");
                Console.WriteLine($"Target: {flatTargets[0]}");
                Console.WriteLine($"VocabSize: {model.Config.VocabSize}");
            }
            
            if (float.IsNaN(batchLoss) || float.IsInfinity(batchLoss))
            {
                Console.WriteLine($"Loss exploded at step {step}");
                break;
            }
            
            if (step % 5 == 0)
                Console.WriteLine($"Step {step}: loss = {batchLoss:F4}");

            tokensTensor.Dispose();
            targetsTensor.Dispose();
            logits.Dispose();
        }
        
        Console.WriteLine("Forward pass complete.");
        Console.WriteLine("Note: Full backward training requires autograd integration - implemented backward methods need to be wired up.");
    }

    private class PixelBatch
    {
        public required int[] Inputs { get; init; }
        public required int[] Targets { get; init; }
    }

    private class PixelCharSource
    {
        public PixelBatch GetBatch(int batchSize)
        {
            var inputs = new int[batchSize * 8];
            var targets = new int[batchSize * 8];

            for (int b = 0; b < batchSize; b++)
            {
                int charBit = (b % 2);
                
                int pixelStart = b * 8;
                for (int p = 0; p < 7; p++)
                    inputs[pixelStart + p] = charBit % 16;  // Ensure < vocabSize
                
                targets[pixelStart + 7] = charBit % 16;
            }

            return new PixelBatch { Inputs = inputs, Targets = targets };
        }
    }
}