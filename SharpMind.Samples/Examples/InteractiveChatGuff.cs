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
    public static class InteractiveChatGuff
    {
        private static readonly string ModelName = "Qwen3-0.6B-Q8_0";
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync()
        {
            CancellationTokenSource cancellationTokenSource = new();
            var ggufPath = Path.Combine(ModelPath, $"{ModelName}.gguf");           
            if (!File.Exists(ggufPath))
            {
                await Console.Out.WriteLineAsync($"GGUF not found: {ggufPath}");
                return;
            }
            GgufLoaderFactory.Default.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);            
            if (tokenizer == null)
            {
                await Console.Out.WriteLineAsync($"Tokenizer dat not found");
                return;
            }
            var sharpConfig = modelConfig.ForModel();

            GC.Collect(); GC.WaitForPendingFinalizers();
            var sw = Stopwatch.StartNew();

            using var weights = GgufLoaderFactory.Default.LoadWeightsToTransformerWeights(ggufPath, modelConfig);
            await Console.Out.WriteLineAsync($"GgufLoader.LoadWeightsToTransformerWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");

            GC.Collect(); GC.WaitForPendingFinalizers();
            sw.Restart();
            using var model = ModelFactory.CreateSession(weights, sharpConfig);
            await Console.Out.WriteLineAsync($"ModelFactory.CreateSession executed in: {sw.Elapsed.TotalSeconds:F2}s");

            sw.Stop();

            await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
            {
                MaxTokens = 512,
                Temperature = 0.7f,
                TopK = 40,
                TopP = 0.9f,
            };
            
            await Console.Out.WriteLineAsync("\nChat ready! Say hello.\n");
            var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);

            async void Response(ChatStreamEntry text)
            {
                Console.ForegroundColor = ConsoleColor.Blue;
                await Console.Out.WriteAsync(text.Token);
            }

            async Task<ChatMessage> Prompt()
            {
                await Console.Out.WriteLineAsync();
                if (!cancellationTokenSource.IsCancellationRequested)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    await Console.Out.WriteAsync($"Prompt:");
                    var prompt = await Console.In.ReadLineAsync() ?? string.Empty;
                    if (prompt == "exit")
                    {
                        cancellationTokenSource.Cancel();
                        return new ChatMessage { Content = prompt, Role = ChatRole.User };
                    }
                    await Console.Out.FlushAsync();
                    await Console.Out.WriteAsync("Response:");
                    return new ChatMessage { Content = prompt, Role = ChatRole.User };
                }
                await Console.Out.WriteLineAsync();
                await Console.Out.FlushAsync();
                cancellationTokenSource.Cancel();
                return new ChatMessage { Content = "exit", Role = ChatRole.User };
            }
        }

    }
}
