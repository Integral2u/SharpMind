using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using SharpMind;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SandBox
{
    public static class LlamaTest
    {
        private static readonly string Assets = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";

        private static void SetupNative()
        {
            // NativeLibraryConfig.All.WithLogCallback(delegate (LLamaLogLevel level, string message) { });
            // NativeLibraryConfig.All.WithAvx(Avx512F.IsSupported ? AvxLevel.Avx512 : Avx2.IsSupported ? AvxLevel.Avx2 : Avx.IsSupported ? AvxLevel.Avx : AvxLevel.None);
        }

        public static async Task TestConcurrency()
        {
            SetupNative();
            CancellationTokenSource cts = new();
            string modelPath = Path.Combine(Assets, "SmolLM-135M.Q4_K_M.gguf");
            string prompt = "hello";
            
            Console.WriteLine($"\n=== Concurrency Test: {Path.GetFileName(modelPath)} ===");
            
            GgufLoader.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
            var sharpConfig = modelConfig.ForModel(HardwareTier.AVX2);
            using var weights = GgufLoader.LoadWeightsToTransformerWeights(modelPath, modelConfig);
            
            int sessionCount = 4;
            var tasks = new List<Task<string>>();
            
            for (int i = 0; i < sessionCount; i++)
            {
                tasks.Add(Task.Run(async () => 
                {
                    var model = ModelFactory.CreateSession(weights, sharpConfig);
                    await using var smSession = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
                    {
                        MaxTokens = 16,
                        Temperature = 0.0f,
                    };
                    
                    var sb = new StringBuilder();
                    await smSession.StartChatAsync(
                        async () => new ChatMessage { Content = prompt, Role = ChatRole.User },
                        async (entry) => 
                        {
                            sb.Append(entry.Token);
                            await Task.CompletedTask;
                        },
                        cts.Token
                    );
                    return sb.ToString();
                }));
            }
            
            var results = await Task.WhenAll(tasks);
            
            bool allMatch = true;
            for (int i = 1; i < results.Length; i++)
            {
                if (results[i] != results[0])
                {
                    Console.WriteLine($"✗ Session {i} result differs from Session 0");
                    Console.WriteLine($"S0: {results[0]}");
                    Console.WriteLine($"S{i}: {results[i]}");
                    allMatch = false;
                }
            }
            
            if (allMatch) Console.WriteLine($"✓ All {sessionCount} sessions produced identical output!");
            else Console.WriteLine("✗ Concurrency test failed: inconsistent outputs.");
        }
    }
}
