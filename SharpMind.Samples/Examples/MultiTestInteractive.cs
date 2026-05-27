using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Samples.Examples
{
    public class MultiTestInteractive

    {
        private static readonly string[] Models =
            [

            "SmolLM-135M.Q4_K_M",
            //"SmolLM2-135M-Instruct.Q4_K_M",
            //"qwen2-0_5b-instruct-q4_k_m",
            //"qwen2-0_5b-instruct-q8_0", //Passing last token delay
            //"qwen2-0_5b-instruct-fp16", //Passing last token delay
            
            
            "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M",
           //"TinyLlama-1.1B-Chat-v1.0.Q4_K_M",
            //"llama-3.2-1b-instruct-q8_0", //Passing last token delay
            //"qwen2.5-1.5b-instruct-q8_0", //Passing last token delay
            
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
                await Console.Out.WriteLineAsync($"Testing {m}");
                await Console.Out.FlushAsync();
                
                GgufLoader.Load(ggufPath, null, out GgufMeta meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                if (tokenizer == null)
                {
                    await Console.Out.WriteLineAsync($"No Tekenizer Data");
                    continue;
                }
                var sharpConfig = modelConfig.ForModel(DetectBestHardware());
                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                var model = ModelFactory.Create(modelConfig, sharpConfig);
                await Console.Out.WriteLineAsync($"ModelFactory.Create executed in: {sw.Elapsed.TotalSeconds:F2}s");

                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();
                GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
                await Console.Out.WriteLineAsync($"GgufLoader.LoadWeightsToModel executed in: {sw.Elapsed.TotalSeconds:F2}s");
                sw.Restart();
                var inferOps = InferenceOpsFactory.Create(sharpConfig, InferenceConfig.Default);
                await Console.Out.WriteLineAsync($"InferenceOpsFactory.Create executed in: {sw.Elapsed.TotalSeconds:F2}s");

                string systemPrompt = "";

                await using var session = new ChatSession(model, tokenizer, inferOps, meta)
                {                     
                    MaxTokens = 32,
                    Temperature = 0.0f,
                    TopK = 1,
                    //TopP = 0.8f
                };
                //session.AddMessage(new ChatMessage() { Content = "You are a polite, creative but methodical AI assistant.", Role = ChatRole.System });
                //session.AddMessage(new ChatMessage() { Content = "Your name is Delta", Role = ChatRole.Agent });
                if (!string.IsNullOrEmpty(systemPrompt)) session.AddMessage(ChatRole.System, systemPrompt);
                var history = await session.StartChatAsync(cancellationTokenSource.Token, Prompt, Response);

                void Response(string text)
                {
                    Console.ForegroundColor = ConsoleColor.Blue;
                    Console.Out.Write(text);
                    tok++;
                    if(tok>15) cancellationTokenSource.Cancel();
                }
                string Prompt()
                {
                    if (!returnedPrompt && !cancellationTokenSource.IsCancellationRequested)
                    {
                        returnedPrompt = true;
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        Console.Out.WriteLine($"Prompt:{prompt}");
                        Console.Out.Flush();
                        Console.Out.Write("Response:");
                        return prompt;
                    }
                    Console.Out.WriteLine();
                    Console.Out.Flush();
                    cancellationTokenSource.Cancel();
                    return "exit";
                }
                await Console.Out.WriteLineAsync();
                await Console.Out.WriteLineAsync($"Tokens per second: {session.TokensPerSecond ?? 0}");
            }
            await Console.Out.WriteLineAsync();
            await Console.Out.WriteLineAsync("Done!");
        }
        private static HardwareTier DetectBestHardware()
        {
            if (Avx2.IsSupported) return HardwareTier.AVX2;
            if (Fma.IsSupported) return HardwareTier.FMA;
            return HardwareTier.Scalar;
        }

    }
}
