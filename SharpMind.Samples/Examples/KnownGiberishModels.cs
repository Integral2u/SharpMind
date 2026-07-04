using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Layers;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace SharpMind.Samples.Examples
{
    public class KnownGiberishModels
    {
        private static readonly string[] Models =
        [            
            "SmolLM-135M.Q4_K_M",           //Response: ctypes initialization Program returnex Sw first customers sheBolplesswith Jointonies predicting
            "SmolLM2-135M-Instruct.Q4_K_M", //Response: indeerymourmereeno?emeteriesccordingquakescessionsrobelinedesidesohyd never $(
            "gemma-3-270m-it-Q8_0",     //Response: incessant Kisan Kisan agron poorest motorway Harareapples kilowattLife economists delivering motorway Highways intensive
            "gemma-3-270m-it-Q4_K_M",   //Response: harmonious?? cheap9 voic goalt llor Arbit privatisation cleats negotiatorsSpar recv justiciaHttpMethod
            "gemma-3-270m-it-F16"       //Response: incessant Kisan Kisan agron poorest HarareImprovingdegenerativeharmonic lousy motorway ProductivityapplesLife intensive                                                
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

                GgufLoader.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                
                Console.Error.WriteLine($"DIAG_CONFIG: arch={modelConfig.Architecture} hidden={modelConfig.HiddenDim} layers={modelConfig.NumLayers} heads={modelConfig.NumHeads} kv={modelConfig.NumKvHeads} ffn={modelConfig.FfnDim} headDim={modelConfig.HeadDim} vocab={modelConfig.VocabSize} rope={modelConfig.RopeTheta}");
                if (tokenizer == null)
                {
                    await Console.Out.WriteLineAsync($"No Tokenizer Data");
                    continue;
                }

                // ── Formatted-prompt diagnostic ──
                var formatter = ChatPromptFormatterFactory.Create(meta);
                if (formatter != null)
                {
                    var testHistory = new List<ChatMessage> { new() { Role = ChatRole.User, Content = "Hello" } };
                    string formatted = formatter.Format(testHistory, tokenizer, addBos: tokenizer.BosId >= 0);
                    Console.Error.WriteLine($"DIAG: formatted prompt: {formatted.Replace("\n", "\\n").Replace("\r", "\\r")}");
                }

                // Dump first layer's weight details
                if (meta.Tensors.Count > 0)
                {
                    for (int ti = 0; ti < Math.Min(20, meta.Tensors.Count); ti++)
                    {
                        var t = meta.Tensors[ti];
                        Console.Error.WriteLine($"DIAG_TENSOR: {t.Name} dtype={t.Dtype} shape=[{string.Join(",", t.Shape)}]");
                    }
                }

                var sharpConfig = modelConfig.ForModel();
                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                using var weights = GgufLoader.LoadWeightsToTransformerWeights(ggufPath, modelConfig);                
                await Console.Out.WriteLineAsync($"GgufLoader.LoadWeightsToTransformerWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();

                using var model = ModelFactory.CreateSession(weights, sharpConfig);
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
            await Console.Out.WriteLineAsync("Done!");
            Console.In.ReadLine();
        }
    }
}