using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind;
using SharpMind.Core.Tensors;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Samples.Examples;

public static class InteractiveChat
{
    private const string GgufFileName = "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf";

    public static Task RunAsync()
    {
        Console.WriteLine("=== SharpMind Interactive Chat ===");
        Console.WriteLine();

        var hardware = DetectBestHardware();
        Console.WriteLine($"Detected Hardware: {hardware}");
        Console.WriteLine();

        var ggufPath = Path.Combine("ExternalAssets", GgufFileName);
        
        long modelVocabSize = 128256;
        long modelHiddenDim = 2048;
        int modelNumLayers = 22;
        int modelNumHeads = 32;
        int modelNumKvHeads = 4;
        int modelFfnDim = 5632;
        int modelMaxSeqLen = 2048;

        if (File.Exists(ggufPath))
        {
            var fileInfo = new FileInfo(ggufPath);
            Console.WriteLine($"GGUF file: {ggufPath}");
            Console.WriteLine($"  Size: {fileInfo.Length / 1_000_000.0:F1} MB");
            
            Console.WriteLine("Attempting GGUF load...");
            try
            {
                var meta = GgufLoader.LoadMeta(ggufPath);
                Console.WriteLine($"  Loaded OK: {meta.TensorCount} tensors");
                
                // Load actual weights
                Console.WriteLine("Loading actual weights...");
                try
                {
                    var weights = GgufLoader.LoadWeights(ggufPath);
                    Console.WriteLine($"  Loaded {weights.Count} weight tensors");
                    
                    // Get model config from GGUF metadata
                    modelVocabSize = meta.GetLong("llama.context_length", 2048);
                    modelHiddenDim = meta.GetLong("llama.embedding_length", 2048);
                    modelNumLayers = (int)meta.GetLong("llama.block_count", 22);
                    modelMaxSeqLen = (int)meta.GetLong("llama.context_length", 2048);
                    Console.WriteLine($"  Config: vocab={modelVocabSize}, hidden={modelHiddenDim}, layers={modelNumLayers}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Weight loading skipped (quantized format): {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  GGUF load failed: {ex.Message}");
            }
        }
        else
        {
            Console.WriteLine($"GGUF not found at: {ggufPath}");
        }
        Console.WriteLine();

        var modelConfig = ModelConfig.Tiny with
        {
            VocabSize = (int)modelVocabSize,
            HiddenDim = (int)modelHiddenDim,
            NumLayers = modelNumLayers,
            NumHeads = modelNumHeads,
            NumKvHeads = modelNumKvHeads,
            FfnDim = modelFfnDim,
            MaxSeqLen = modelMaxSeqLen,
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

        Console.WriteLine("Building model with JigSaw kernels...");
        
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        
        Console.WriteLine($"Model built: {model.ParameterCount / 1_000_000.0:F1}M parameters");
        Console.WriteLine();

        Console.WriteLine("=== Forward Pass Test ===");
        
        using var input = Tensor<int>.From([1], 1, 1);
        using var logits = model.Forward(input);
        
        Console.WriteLine($"Input: [1, 1] token");
        Console.WriteLine($"Logits: [{logits.Shape.Rows}, {logits.Shape.Cols}, {logits.Shape[2]}]");
        
        var vocabSize = logits.Shape[2];
        var slice = logits.Data.Slice(0, Math.Min(20, vocabSize)).ToArray();
        
        var topIndices = Enumerable.Range(0, slice.Length)
            .OrderByDescending(i => slice[i])
            .Take(5)
            .ToList();
            
        Console.WriteLine("Top 5 token scores (random init):");
        foreach (var i in topIndices)
        {
            Console.WriteLine($"  id:{i} = {slice[i]:F2}");
        }

        Console.WriteLine();
        Console.WriteLine("Demo complete!");
        Console.WriteLine("Note: Model uses random weights - need trained GGUF weights for chat.");
        
        return Task.CompletedTask;
    }

    private static HardwareTier DetectBestHardware()
    {
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (Fma.IsSupported) return HardwareTier.FMA;
        return HardwareTier.Scalar;
    }
}