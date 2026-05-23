using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Samples.Examples
{
    public class MultiTestInteractive

    {
        private static readonly string[] Models =
            [
            "qwen2-0_5b-instruct-q8_0",
            "llama-3.2-1b-instruct-q8_0",
            "qwen2.5-1.5b-instruct-q8_0",
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M"
            ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt)
        {
            
            foreach (var m in Models)
            {
                var returnedPrompt = false;
                var tok = 0;
                CancellationTokenSource cancellationTokenSource = new();
                var ggufPath = Path.Combine(ModelPath, $"{m}.gguf");
                var tokenizerPath = Path.Combine(ModelPath, $"{m}.json");
                if (!File.Exists(ggufPath)) continue;
                Console.Out.WriteLine($"Testing {m}");
                Console.Out.Flush();
                GgufLoader.Load(ggufPath, null, out GgufMeta meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                if (tokenizer == null)
                {
                    Console.Out.WriteLine($"No Tekenizer Data");
                    continue;
                }
                var sharpConfig = modelConfig.ForModel(DetectBestHardware());

                GC.Collect(); GC.WaitForPendingFinalizers();
                var model = ModelFactory.Create(modelConfig, sharpConfig);
                GC.Collect(); GC.WaitForPendingFinalizers();
                GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
                var inferOps = InferenceOpsFactory.Create(sharpConfig, InferenceConfig.Default);
                
                string systemPrompt = "";

                await using var session = new ChatSession(model, tokenizer, inferOps, meta)
                {
                    MaxTokens = 32,
                    Temperature = 0.0f,
                    TopK = 1,
                };
                if (!string.IsNullOrEmpty(systemPrompt)) session.AddMessage(ChatRole.System, systemPrompt);
                var history = await session.StartChatAsync(cancellationTokenSource.Token, Prompt, Response);
                Console.Out.WriteLine();
                void Response(string text)
                {
                    Console.ForegroundColor = ConsoleColor.Blue;
                    Console.Write(text);
                    tok++;
                    if(tok>15) cancellationTokenSource.Cancel();
                }

                string Prompt()
                {
                    if (!returnedPrompt && !cancellationTokenSource.IsCancellationRequested)
                    {
                        returnedPrompt = true;
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        Console.Out.WriteLine(prompt);
                        Console.Out.Flush();
                        return prompt;
                    }
                    Console.Out.WriteLine();
                    Console.Out.Flush();
                    cancellationTokenSource.Cancel();
                    return "exit";
                }
            }
            Console.Out.WriteLine();
            Console.Out.WriteLine("Done!");
        }
        private static HardwareTier DetectBestHardware()
        {
            if (Avx2.IsSupported) return HardwareTier.AVX2;
            if (Fma.IsSupported) return HardwareTier.FMA;
            return HardwareTier.Scalar;
        }

    }
}
