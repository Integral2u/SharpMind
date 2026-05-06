using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;

namespace SharpMind.Samples.Tests;

/// <summary>
/// Loads TinyLlama from GGUF and runs a chat session demo.
/// </summary>
public static class TinyLlamaChat
{
    public static Task RunAsync()
    {
        Console.WriteLine("=== TinyLlama ChatSession Demo ===");
        Console.WriteLine();

        // Use TinyLlama config directly
        var sharpConfigModel = new Model.Format.SharpMindConfig
        {
            VocabSize = 128256,
            HiddenDim = 2048,
            NumLayers = 22,
            NumHeads = 32,
            NumKvHeads = 4,
            FfnDim = 5632,
            MaxSeqLen = 2048,
            RopeTheta = 500000f,
            Architecture = "decoder",
            Activation = "silu",
            Gate = "swiglu",
            Ffn = "gated",
            Norm = "rmsnorm",
            Attention = "gqa",
        };
        
        Console.WriteLine("Using TinyLlama-1.1B-Chat config:");
        Console.WriteLine($"  VocabSize: {sharpConfigModel.VocabSize}");
        Console.WriteLine($"  HiddenDim: {sharpConfigModel.HiddenDim}");
        Console.WriteLine($"  NumLayers: {sharpConfigModel.NumLayers}");
        Console.WriteLine($"  NumHeads: {sharpConfigModel.NumHeads}");
        Console.WriteLine($"  MaxSeqLen: {sharpConfigModel.MaxSeqLen}");
        Console.WriteLine();

        // Create model config
        var modelConfig = sharpConfigModel.ToModelConfig();

        // Build JigSaw config
        var sharpConfig = sharpConfigModel.ToJigSawConfig();
        
        // Force scalar hardware for compatibility
        var sharpConfigWithHardware = sharpConfig with { Hardware = HardwareTier.Scalar };

        // Build transformer (random weights)
        Console.WriteLine("Building Transformer...");
        var model = ModelFactory.Create(modelConfig, sharpConfigWithHardware);
        
        Console.WriteLine($"Model: {model.ParameterCount / 1_000_000.0:F1}M parameters");
        Console.WriteLine();

        // Test the generation flow
        TestGenerationFlow(model);

        Console.WriteLine();
        Console.WriteLine("Demo complete!");
        
        return Task.CompletedTask;
    }

    private static void TestGenerationFlow(Transformer model)
    {
        Console.WriteLine("Testing model forward pass...");
        
        // Create dummy input (single token)
        using var input = Tensor<int>.From([1], 1, 1);
        
        // Get logits
        using var logits = model.Forward(input);
        
        Console.WriteLine($"Input shape: [{input.Shape.Rows}, {input.Shape.Cols}]");
        Console.WriteLine($"Logits shape: [{logits.Shape.Rows}, {logits.Shape.Cols}, {logits.Shape[2]}]");
        Console.WriteLine($"Vocab size: {logits.Shape[2]}");
        
        // Sample a token
        var lastLogits = logits.Data[..logits.Shape[2]].ToArray();
        var sampled = Array.IndexOf(lastLogits, lastLogits.Max());
        
        Console.WriteLine($"Sampled token ID: {sampled}");
        Console.WriteLine("Forward pass successful!");
    }
}