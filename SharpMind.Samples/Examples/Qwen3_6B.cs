using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Model.Layers;
using SharpMind.Model.Layers.Attention;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SharpMind.Samples.Examples
{
    public class Qwen3_6B
    {
        //https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/tree/main
        private static readonly string[] Models =
        [
            "Qwen3-0.6B-Q8_0",      //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q6_K",      //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q5_K_M",    //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q4_K_M",    //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q4_1",      //Response:
            "Qwen3-0.6B-Q4_0",      //Response:<think>\nOkay, the user just said "Hello," so I need to
            "Qwen3-0.6B-Q3_K_M",    //Response:<think>\nOkay, the user is asking for help with a problem. But
            "Qwen3-0.6B-Q2_K",      //Response:?\nOkay, so I need to start with the user's message.
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
                if (!File.Exists(ggufPath)) continue;
                await Console.Out.WriteLineAsync($"Testing {m}");
                await Console.Out.FlushAsync();

                GgufLoaderFactory.Default.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                if (tokenizer == null)
                {
                    await Console.Out.WriteLineAsync($"No Tokenizer Data");
                    continue;
                }

                var sharpConfig = modelConfig.ForModel();
                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                using var weights = GgufLoaderFactory.Default.LoadWeightsToTransformerWeights(ggufPath, modelConfig);
                await Console.Out.WriteLineAsync($"GgufLoader.LoadWeightsToTransformerWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();
                // ── Formatted-prompt diagnostic ──
                var formatter = ChatPromptFormatterFactory.Create(meta);
                if (formatter != null)
                {
                    var testHistory = new List<ChatMessage> { new() { Role = ChatRole.User, Content = prompt } };
                    string formatted = formatter.Format(testHistory, tokenizer, addBos: tokenizer.BosId >= 0);
                    Console.Error.WriteLine($"DIAG: formatted prompt: {formatted.Replace("\n", "\\n").Replace("\r", "\\r")}");
                }
                using var model = ModelFactory.CreateSession(weights, sharpConfig);
                await Console.Out.WriteLineAsync($"ModelFactory.CreateSession executed in: {sw.Elapsed.TotalSeconds:F2}s");

                sw.Stop();

                await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
                {
                    MaxTokens = 50,
                    Temperature = 0.0f,
                    TopK = 1,
                };
                var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);

                async void Response(ChatStreamEntry entry)
                {
                    Console.ForegroundColor = ConsoleColor.Blue;
                    if (entry.TokenId.HasValue)
                        Console.Error.Write($"[{entry.TokenId.Value}]");
                    await Console.Out.WriteAsync(entry.Token);
                    tok++;
                    if (tok > 50) cancellationTokenSource.Cancel();
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
        }
    }
}
