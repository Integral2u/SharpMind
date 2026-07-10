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
        public static float CurrentProgress = 0; 
        public static void Progress(float p)
        {
            if (CurrentProgress < p && CurrentProgress < 0.91f)
            {
                CurrentProgress += 0.1f;
                Console.Write(".");
            }
            if(CurrentProgress > 0.91f)
            {
                CurrentProgress = 0f;
                Console.WriteLine();
            }
        }
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
                // Build the JigSaw mapping with CPU baseline, then override
                // activations and gates to use GPU-accelerated kernels (via WithGpu()).
                // JigSaw's external [PuzzlePeice] scan finds the GPU kernels because
                // SharpMind.GPU is loaded in the AppDomain (WithGpu() lives there).
                var mapping = withGPU ? new MappingBuilder()
                    .ApplyPreset(sharpConfig)
                    .WithGpu()
                    .Build() : null;
                Dictionary<string, string> qOpsMapping = (new QuantizationConfig { Hardware = sharpConfig.Hardware }).ToJigSawMapping();

                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                var loader = new GgufLoader(QuantizationFactory.Create(qOpsMapping), ggufPath, modelConfig, LoadMode.Full);
                
                using var weights = loader.LoadWeightsToTransformerWeights(new Progress<float>(p=> Progress(p)));
                
                await Console.Out.WriteLineAsync($"GgufLoader.LoadWeightsToTransformerWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
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
