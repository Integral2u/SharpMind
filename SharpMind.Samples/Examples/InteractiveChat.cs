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

        var sharpConfigModel = CreateTinyLlamaConfig();
        Console.WriteLine($"Model: TinyLlama-1.1B-Chat ({sharpConfigModel.NumLayers} layers, {sharpConfigModel.HiddenDim} hidden)");
        Console.WriteLine();

        Console.WriteLine("Building model with optimal kernels...");
        var modelConfig = sharpConfigModel.ToModelConfig();
        
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

        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Console.WriteLine($"Model parameters: {model.ParameterCount / 1_000_000.0:F1}M");
        Console.WriteLine();

        Console.WriteLine("=== Interactive Generation ===");
        Console.WriteLine("Note: Using random weights - output will be noise");
        Console.Write("Enter prompt (or press Enter for default): ");
        var input = Console.ReadLine();
        
        if (string.IsNullOrWhiteSpace(input))
            input = "Hello";
        
        Console.WriteLine($"Input: {input}");
        
        // Simple token generation using greedy sampling
        Console.WriteLine("\nTesting single forward pass...");
        
        using var testInput = Tensor<int>.From([1], 1, 1);
        using var testLogits = model.Forward(testInput);
        
        Console.WriteLine($"Logits shape: [{testLogits.Shape.Rows}, {testLogits.Shape.Cols}, {testLogits.Shape[2]}]");
        
        var slice = testLogits.Data.Slice(0, Math.Min(10, testLogits.Shape[2])).ToArray();
        Console.WriteLine($"First 10 logits: [{string.Join(", ", slice.Select(x => x.ToString("F2")))}]");
        
        Console.WriteLine("\nDemo complete!");
        
        return Task.CompletedTask;
    }

    private static int[] GenerateTokens(Transformer model, int inputLen)
    {
        var generated = new List<int>();
        var input = Tensor<int>.From([1], 1, 1);
        
        try
        {
            // Just generate 3 tokens for demo
            for (int i = 0; i < 3; i++)
            {
                using var logits = model.Forward(input);
                
                // Greedy sample
                var vocabSize = logits.Shape[2];
                var slice = logits.Data.Slice(0, vocabSize).ToArray();
                int next = Array.IndexOf(slice, slice.Max());
                
                generated.Add(next);
                Console.Write(".");
                
                // Use the sampled token as next input
                input.Dispose();
                input = Tensor<int>.From([next], 1, 1);
            }
        }
        finally
        {
            input.Dispose();
        }
        
        return generated.ToArray();
    }

    private static HardwareTier DetectBestHardware()
    {
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (Fma.IsSupported) return HardwareTier.FMA;
        return HardwareTier.Scalar;
    }

    private static Model.Format.SharpMindConfig CreateTinyLlamaConfig()
    {
        return new Model.Format.SharpMindConfig
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
    }
}