using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using Xunit;

namespace SharpMind.Samples.Tests;

/// <summary>
/// Loads TinyLlama from GGUF and runs a chat session demo.
/// </summary>
public class TinyLlamaChatTests
{
    [Fact]
    public void RunDemo()
    {
        RunAsync().Wait();
    }
    
    public Task RunAsync()
    {
        Console.WriteLine("=== TinyLlama ChatSession Demo ===");
        var ggufPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\qwen2-0_5b-instruct-q4_k_m.gguf";
        var meta = GgufLoader.LoadMeta(ggufPath);
        
        foreach (var t in meta.Tensors.Take(10))
        {
            Console.WriteLine($"  Tensor: {t.Name}, Shape: [{string.Join(",", t.Shape)}]");
        }

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