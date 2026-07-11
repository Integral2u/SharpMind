using SharpMind.Core.Quantization;
using SharpMind.GPU;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SharpMind.Samples.Examples
{
    public class ModelListRunner
    {        
        public static async Task RunAsync(string prompt, string ModelPath, string[] Models, bool withGPU = false)
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

                GgufLoader.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                if (tokenizer == null)
                {
                    await Console.Out.WriteLineAsync($"No Tokenizer Data");
                    continue;
                }

                var sharpConfig = modelConfig.ForModel();
                // Build a single combined mapping. WithGpu() now overrides quant ops
                // as well as model-level ops — no separate qOpsMapping needed.
                var mapping = withGPU ? new MappingBuilder(sharpConfig.ResolvedHardware)
                    .ApplyPreset(sharpConfig)
                    .ApplyQuantPreset(sharpConfig)
                    .WithGpu()
                    .Build() : null;

                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                var qOps = QuantizationFactory.Create(mapping ?? sharpConfig.ToJigSawMapping());
                using var weights = ModelFactory.Create(modelConfig, sharpConfig, qOps, ggufPath, meta, LoadMode.Full);
                weights.InitializeWeights();
                
                await Console.Out.WriteLineAsync($"ModelFactory.Create + InitializeWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();
                using var model = ModelFactory.CreateSession(weights, sharpConfig, mapping);
                await Console.Out.WriteLineAsync($"ModelFactory.CreateSession executed in: {sw.Elapsed.TotalSeconds:F2}s");

                sw.Stop();

                await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
                {
                    MaxTokens = 256,
                    Temperature = 0.0f,
                    TopK = 1,
                };
                try
                {
                    var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);
                }
                catch (Exception ex)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    await Console.Out.WriteLineAsync();
                    await Console.Out.WriteLineAsync($"EXCEPTION: {ex.GetType().Name}: {ex.Message}");
                    await Console.Out.WriteLineAsync(ex.StackTrace?[..500] ?? "(no stack)");
                }

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
        }
    }
}
