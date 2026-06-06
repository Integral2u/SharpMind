using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpMind.Samples.Examples
{
    public class QwenQ8
    {
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        private static readonly string Model = "qwen2-0_5b-instruct-q8_0";
        public static async Task RunAsync(string prompt)
        {
            Console.ForegroundColor = ConsoleColor.White;
            var returnedPrompt = false;
            var tok = 0;
            CancellationTokenSource cancellationTokenSource = new();
            var ggufPath = Path.Combine(ModelPath, $"{Model}.gguf");
            var tokenizerPath = Path.Combine(ModelPath, $"{Model}.json");
            if (!File.Exists(ggufPath))
            {
                await Console.Out.WriteLineAsync($"{Model}.gguf not found.");
                Console.In.ReadLine();
                return;
            }
            await Console.Out.WriteLineAsync($"Testing {Model}");
            await Console.Out.FlushAsync();

            GgufLoader.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
            if (tokenizer == null)
            {
                await Console.Out.WriteLineAsync($"No Tokenizer Data");
                Console.In.ReadLine();
                return;
            }

            var sharpConfig = modelConfig.ForModel();
            GC.Collect(); GC.WaitForPendingFinalizers();
            var sw = Stopwatch.StartNew();


            var model = ModelFactory.Create(modelConfig, sharpConfig);
            await Console.Out.WriteLineAsync($"ModelFactory.Create executed in: {sw.Elapsed.TotalSeconds:F2}s");

            GC.Collect(); GC.WaitForPendingFinalizers();
            sw.Restart();
            GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
            await Console.Out.WriteLineAsync($"GgufLoader.LoadWeightsToModel executed in: {sw.Elapsed.TotalSeconds:F2}s");

            sw.Stop();

            await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
            {
                MaxTokens = 256,
                Temperature = 0.0f,
                TopK = 1,
            };
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

            await Console.Out.WriteLineAsync();
            await Console.Out.WriteLineAsync("Done!");
            Console.In.ReadLine();
        }
    }
}
