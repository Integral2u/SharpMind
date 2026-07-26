using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SharpMind.Samples.Examples
{
    public class BuilderOptions
    {
        public static async Task RunAsync(string prompt, string ModelPath, string ModelName)
        {
            Type[] cacheBuilders = [typeof(QuantizedKVCacherBuilder), typeof(PagedKVCacherBuilder),typeof(KVCacherBuilder)];
            Type[] generatorBuilders = [typeof(StandardGeneratorBuilder<>),typeof(MedusaGeneratorBuilder<>),typeof(SpeculativeGeneratorBuilder<>)];
            string modelPath = string.Empty;
            ModelFormat? fmt = null;
            foreach (var mFmt in Enum.GetValues<ModelFormat>()) {
                var ext = ModelFormatHelpers.GetExtension(mFmt);
                modelPath = Path.Combine(ModelPath, $"{ModelName}{ext}");
                if (File.Exists(modelPath))
                {
                    fmt = mFmt; break;
                }                
            }
            if (fmt == null)
            {
                await Console.Out.WriteLineAsync($"{ModelName} not found.");
                Console.In.ReadLine();
                return;
            }
            var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);
            metaHelper.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);

            if (tokenizer == null)
            {
                await Console.Out.WriteLineAsync($"No Tokenizer Data");
                Console.In.ReadLine();
                return;
            }
            var sharpConfig = modelConfig.ForModel();

            GC.Collect(); GC.WaitForPendingFinalizers();
            var sw = Stopwatch.StartNew();
            var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
            using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath);
            weights.InitializeWeights();
            await Console.Out.WriteLineAsync($"ModelFactory.Create + InitializeWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
            

            foreach (var generatorDef in generatorBuilders)
            {

                foreach (var cacheBuilder in cacheBuilders)
                {
                    Console.ForegroundColor = ConsoleColor.White;
                    var returnedPrompt = false;
                    var tok = 0;
                    CancellationTokenSource cancellationTokenSource = new();
                    
                    await Console.Out.WriteLineAsync($"Testing {ModelName} using {generatorDef.Name},{cacheBuilder}");
                    await Console.Out.FlushAsync();
                    GC.Collect(); GC.WaitForPendingFinalizers();
                    sw.Restart();
                    using var model = ModelFactory.CreateTransformer(weights, sharpConfig);
                    await Console.Out.WriteLineAsync($"ModelFactory.CreateTransformer executed in: {sw.Elapsed.TotalSeconds:F2}s");

                    sw.Reset();

                    await using var session = ChatSessionFactory.CreateChatSession(generatorDef, cacheBuilder, model, tokenizer, meta);                   
                    session.MaxTokens = 256;
                    session.Temperature = 0.0f;
                    session.TopK = 1;
                    session.InitializeChat();
                    await Console.Out.WriteLineAsync($"ChatSession executed in: {sw.Elapsed.TotalSeconds:F2}s");
                    sw.Stop();
                    async void Response(ChatStreamEntry text)
                    {
                        Console.ForegroundColor = text.Status == ChatStatus.Thinking ? ConsoleColor.Gray : ConsoleColor.Blue;
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
