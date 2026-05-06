using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;

namespace SharpMind.Samples.Examples;

public static class InteractiveChat
{
    public static Task RunAsync()
    {
        Console.WriteLine("=== SharpMind Interactive Chat ===");
        Console.WriteLine();

        var hardware = DetectBestHardware();
        Console.WriteLine($"Detected Hardware: {hardware}");
        Console.WriteLine();

        // TinyLlama config from known values
        var modelConfig = ModelConfig.Tiny with
        {
            VocabSize = 128256,
            HiddenDim = 2048,
            NumLayers = 22,
            NumHeads = 32,
            NumKvHeads = 4,
            FfnDim = 5632,
            MaxSeqLen = 2048,
        };

        var sharpConfig = new SharpMindConfig
        {
            Activation = ActivationKind.SiLU,
            Gate = GateKind.SwiGLU,
            Ffn = FfnKind.Gated,
            Attention = AttentionKind.GQA,
            Norm = NormKind.RMSNorm,
            Arch = ArchKind.Decoder,
            Hardware = hardware,
        };

        Console.WriteLine("Model: TinyLlama-1.1B-Chat");
        Console.WriteLine($"  VocabSize: {modelConfig.VocabSize}");
        Console.WriteLine($"  HiddenDim: {modelConfig.HiddenDim}");
        Console.WriteLine($"  NumLayers: {modelConfig.NumLayers}");
        Console.WriteLine($"  NumHeads: {modelConfig.NumHeads}");
        Console.WriteLine($"  MaxSeqLen: {modelConfig.MaxSeqLen}");
        Console.WriteLine();

        Console.WriteLine("Building model...");
        
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        
        Console.WriteLine($"Model parameters: {model.ParameterCount / 1_000_000.0:F1}M");
        Console.WriteLine();

        Console.WriteLine("=== Generation Test ===");
        Console.Write("Enter prompt: ");
        
        // Simple forward pass test
        using var input = Tensor<int>.From([1], 1, 1);
        using var logits = model.Forward(input);
        
        Console.WriteLine($"Logits shape: [{logits.Shape.Rows}, {logits.Shape.Cols}, {logits.Shape[2]}]");
        
        // Get top tokens
        var vocabSize = logits.Shape[2];
        var slice = logits.Data.Slice(0, Math.Min(100, vocabSize)).ToArray();
        
        var topIndices = Enumerable.Range(0, slice.Length)
            .OrderByDescending(i => slice[i])
            .Take(5)
            .ToList();
            
        Console.WriteLine("Top 5 token logits:");
        foreach (var i in topIndices)
        {
            Console.WriteLine($"  Token {i}: {slice[i]:F2}");
        }

        Console.WriteLine();
        Console.WriteLine("Note: Model uses random weights (not trained).");
        Console.WriteLine("GGUF loading in progress - need trained weights for chat.");
        Console.WriteLine("\nDemo complete!");
        
        return Task.CompletedTask;
    }

    private static HardwareTier DetectBestHardware()
    {
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (Fma.IsSupported) return HardwareTier.FMA;
        return HardwareTier.Scalar;
    }
}