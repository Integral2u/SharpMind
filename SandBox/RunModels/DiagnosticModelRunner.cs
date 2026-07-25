using SharpMind.Core.Quantization;
using SharpMind.GPU;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SandBox.RunModels
{
    public class DiagnosticModelRunner
    {
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync(string prompt, string[] Models, bool withGPU = false)
        {
            var totalTime = Stopwatch.StartNew();
            foreach (var m in Models)
            {
                Console.ForegroundColor = ConsoleColor.White;
                var returnedPrompt = false;
                var tok = 0;
                CancellationTokenSource cancellationTokenSource = new();
                string modelPath = string.Empty;
                ModelFormat? fmt = null;
                foreach (var mFmt in Enum.GetValues<ModelFormat>())
                {
                    var ext = ModelFormatHelpers.GetExtension(mFmt);
                    modelPath = Path.Combine(ModelPath, $"{m}{ext}");
                    if (File.Exists(modelPath))
                    {
                        fmt = mFmt; break;
                    }
                }
                if (fmt == null) continue;
                var metaHelper = ModelFormatHelpers.GetModelMetaHelperFor((ModelFormat)fmt);

                await Console.Out.WriteLineAsync($"Testing {m}");

                await Console.Out.FlushAsync();
                metaHelper.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                Console.Out.WriteLineAsync($"[DIAG] {meta}: arch='{meta.GetString("general.architecture")}'");
                // Diagnostic: dump rendered prompt and formatter errors
                var formatter = ChatPromptFormatterFactory.Create(meta);
                if (formatter is JinjaTemplateFormatter jinja)
                {
                    var testMsgs = new[] { ChatMessage.User(prompt) };
                    string rendered = jinja.Format(testMsgs, tokenizer, false);
                    Console.Error.WriteLine($"[formatter] prompt={rendered.Replace("\n", "\\n")}");
                    if (jinja.LastErrors is { Count: > 0 })
                        foreach (var err in jinja.LastErrors)
                            Console.Error.WriteLine($"[formatter ERROR] {err}");
                }
                if (tokenizer == null)
                {
                    await Console.Out.WriteLineAsync($"No Tokenizer Data");
                    continue;
                }

                var sharpConfig = modelConfig.ForModel();
                // Build a single combined mapping. WithGpu() now overrides quant ops
                // as well as model-level ops — no separate qOpsMapping needed.
                var mapping = withGPU ? new SharpMind.MappingBuilder(sharpConfig.ResolvedHardware)
                    .ApplyPreset(sharpConfig)
                    .ApplyQuantPreset(sharpConfig)
                    .WithGpu()
                    .Build() :
                    new SharpMind.MappingBuilder(sharpConfig.ResolvedHardware)
                    .ApplyPreset(sharpConfig)
                    .ApplyQuantPreset(sharpConfig)
                    .Build();

                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                var qOps = QuantizationFactory.Create(mapping);
                using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath);
                weights.InitializeWeights();

                await Console.Out.WriteLineAsync($"ModelFactory.Create + InitializeWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();
                using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
                await Console.Out.WriteLineAsync($"ModelFactory.CreateTransformer executed in: {sw.Elapsed.TotalSeconds:F2}s");

                sw.Restart();

                await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
                {
                    MaxTokens = 256,
                    Temperature = 0.0f,
                    TopK = 1,
                };
                await Console.Out.WriteLineAsync($"ChatSession executed in: {sw.Elapsed.TotalSeconds:F2}s");
                sw.Stop();
                var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);


                async void Response(ChatStreamEntry text)
                {
                    Console.ForegroundColor = ConsoleColor.Blue;
                    await Console.Out.WriteAsync(text.Token);
                    tok++;
                    if (tok > 25) cancellationTokenSource.Cancel();
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
            await Console.Out.WriteLineAsync($"All Models Executed in: {totalTime.Elapsed.TotalSeconds:F2}s");
            await Console.Out.WriteLineAsync();
        }
    }
}
