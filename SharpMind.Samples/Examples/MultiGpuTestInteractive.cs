using SharpMind.GPU;
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
    public class MultiGpuTestInteractive

    {
        private static readonly string[] Models =
            [
            "SmolLM-135M.Q4_K_M",
            "qwen2-0_5b-instruct-q8_0",
            ];

        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt)
        {

            foreach (var m in Models)
            {
                Console.ForegroundColor = ConsoleColor.White;
                var returnedPrompt = false;
                var tok = 0;
                CancellationTokenSource cancellationTokenSource = new();
                var ggufPath = Path.Combine(ModelPath, $"{m}.gguf");
                var tokenizerPath = Path.Combine(ModelPath, $"{m}.json");
                if (!File.Exists(ggufPath)) continue;
                await Console.Out.WriteLineAsync($"Testing {m} (GPU kernels)");
                await Console.Out.FlushAsync();

                GgufLoader.Load(ggufPath, null, out GgufMeta meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                if (tokenizer == null)
                {
                    await Console.Out.WriteLineAsync($"No Tokenizer Data");
                    continue;
                }

                var sharpConfig = modelConfig.ForModel(DetectBestHardware());
                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();

                // Build the JigSaw mapping with CPU baseline, then override
                // activations and gates to use GPU-accelerated kernels (via WithGpu()).
                // JigSaw's external [PuzzlePeice] scan finds the GPU kernels because
                // SharpMind.GPU is loaded in the AppDomain (WithGpu() lives there).
                var mapping = new MappingBuilder(DetectBestHardware())
                    .ApplyPreset(sharpConfig)
                    .WithGpu()
                    .Build();

                var model = ModelFactory.Create(modelConfig, sharpConfig, mapping);
                await Console.Out.WriteLineAsync($"ModelFactory.Create (GPU) executed in: {sw.Elapsed.TotalSeconds:F2}s");

                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();
                GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
                string systemPrompt = "";

                var generator = new StandardGenerator(model, tokenizer);
                await using var session = new ChatSession(generator, tokenizer, meta)
                {
                    MaxTokens = 256,
                    Temperature = 0.0f,
                    TopK = 1,
                };
                if (!string.IsNullOrEmpty(systemPrompt)) session.AddMessage(ChatRole.System, systemPrompt);
                var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);

                async void Response(ChatStreamEntry text)
                {
                    Console.ForegroundColor = ConsoleColor.Blue;
                    await Console.Out.WriteAsync(text.Token);
                    tok++;
                    if (tok > 15) cancellationTokenSource.Cancel();
                }
                async Task<ChatMessage> Prompt()
                {
                    if (!returnedPrompt && !cancellationTokenSource.IsCancellationRequested)
                    {
                        returnedPrompt = true;
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        await Console.Out.WriteLineAsync($"Prompt:{prompt}");
                        await Console.Out.FlushAsync();
                        await Console.Out.WriteAsync("Response:");
                        return new ChatMessage { Content = prompt, Role = ChatRole.User };
                    }
                    await Console.Out.WriteLineAsync();
                    await Console.Out.FlushAsync();
                    cancellationTokenSource.Cancel();
                    return new ChatMessage { Content = "exit", Role = ChatRole.User };
                }
                await Console.Out.WriteLineAsync();
                await Console.Out.WriteLineAsync($"Tokens per second: {session.TokensPerSecond ?? 0:F2}  TTFT: {session.TimeToFirstToken?.ToString("F3") ?? "N/A"}s");
            }
            await Console.Out.WriteLineAsync();
            await Console.Out.WriteLineAsync("Done!");
            Console.In.ReadLine();
        }

        private static HardwareTier DetectBestHardware()
        {
            if (Avx2.IsSupported) return HardwareTier.AVX2;
            if (Fma.IsSupported) return HardwareTier.FMA;
            return HardwareTier.Scalar;
        }

    }
}
