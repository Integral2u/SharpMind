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
    public class BuilderOptions
    {
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        private static readonly string Model = "qwen2-0_5b-instruct-q8_0";
        public static async Task RunAsync(string prompt)
        {
            Type[] cacheBuilders = [typeof(KVCacherBuilder), typeof(PagedKVCacherBuilder)];
            Type[] generatorBuilders = [typeof(StandardGeneratorBuilder<>), typeof(SpeculativeGeneratorBuilder<>)];

            var ggufPath = Path.Combine(ModelPath, $"{Model}.gguf");
            if (!File.Exists(ggufPath))
            {
                await Console.Out.WriteLineAsync($"{Model}.gguf not found.");
                Console.In.ReadLine();
                return;
            }

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

            foreach (var generatorDef in generatorBuilders)
            {

                foreach (var cacheBuilder in cacheBuilders)
                {
                    Console.ForegroundColor = ConsoleColor.White;
                    var returnedPrompt = false;
                    var tok = 0;
                    CancellationTokenSource cancellationTokenSource = new();
                    
                    await Console.Out.WriteLineAsync($"Testing {Model} using {generatorDef.Name},{cacheBuilder}");
                    await Console.Out.FlushAsync();
                    sw.Stop();
                    
                    await using var session = ChatSessionFactory.CreateChatSession(generatorDef, cacheBuilder, model, tokenizer, meta);                   
                    session.MaxTokens = 256;
                    session.Temperature = 0.0f;
                    session.TopK = 1;

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

                    var history = await session.StartChatAsync( Prompt,Response,cancellationTokenSource.Token);

                    await Console.Out.WriteLineAsync();
                    await Console.Out.WriteLineAsync($"Tokens per second: {session.TokensPerSecond ?? 0:F2}  TTFT: {session.TimeToFirstToken?.ToString("F3") ?? "N/A"}s");
                }
            }
            await Console.Out.WriteLineAsync();
            await Console.Out.WriteLineAsync("Done!");
            Console.In.ReadLine();
        }
    }
}
