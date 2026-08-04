using SharpMind.Core;
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
using System.Runtime.Serialization;

namespace SharpMind.Samples.Examples
{
    public class ModelListRunner
    {
        /// <summary>
        /// Convenience overload testing a single prompt against every model.
        /// </summary>
        public static Task RunAsync(string prompt, string ModelPath, string[] Models, bool withGPU = false, int maxTokens = 100, LoadMode loadMode = LoadMode.Full, IChatPromptFormatter? formatter = null)
            => RunAsync(new[] { prompt }, ModelPath, Models, withGPU, maxTokens, loadMode, formatter);

        /// <summary>
        /// Loads each model once, then runs every <paramref name="prompts"/> against
        /// it in a <em>fresh</em> session (new KV cache + history) before loading the
        /// next model. When <paramref name="formatter"/> is null the formatter is
        /// resolved per model from the chat template stored in its file (via
        /// <see cref="ChatPromptFormatterFactory.Create(ModelMetaData?, Tokenizer)"/>),
        /// so each freshly loaded model formats its own prompts correctly.
        /// </summary>
        public static async Task RunAsync(string[] prompts, string ModelPath, string[] Models, bool withGPU = false, int maxTokens = 100, LoadMode loadMode = LoadMode.Full, IChatPromptFormatter? formatter = null)
        {
            if (prompts is not { Length: > 0 }) return;
            var totalTime = Stopwatch.StartNew();
            foreach (var m in Models)
            {
                Console.ForegroundColor = ConsoleColor.White;
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
                    .Build() :
                    new MappingBuilder(sharpConfig.ResolvedHardware)
                    .ApplyPreset(sharpConfig)
                    .ApplyQuantPreset(sharpConfig)
                    .Build();

                GC.Collect(); GC.WaitForPendingFinalizers();
                var sw = Stopwatch.StartNew();
                var qOps = QuantizationFactory.Create(mapping);
                using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath, loadMode);
                weights.InitializeWeights();

                await Console.Out.WriteLineAsync($"ModelFactory.Create + InitializeWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");
                GC.Collect(); GC.WaitForPendingFinalizers();
                sw.Restart();
                using var model = ModelFactory.CreateTransformer(weights, sharpConfig, mapping);
                await Console.Out.WriteLineAsync($"ModelFactory.CreateTransformer executed in: {sw.Elapsed.TotalSeconds:F2}s");

                // One shared formatter for this model — auto-resolved from the
                // stored chat template (Jinja/ChatML) when none was supplied.
                var resolvedFormatter = formatter ?? ChatPromptFormatterFactory.Create(meta, tokenizer);

                // Load once, then test every prompt in its own fresh session.
                foreach (var prompt in prompts)
                {
                    sw.Restart();
                    // disposeModel:false keeps the model alive so later prompts
                    // reuse it; the outer `using var model` frees it afterwards.
                    await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta, null, null, null, null, null, null, resolvedFormatter, disposeModel: false)
                    {
                        MaxTokens = 256,
                        Temperature = 0.6f,
                        TopK = 40,
                    };
                    session.InitializeChat();

                    var history = await RunPromptAsync(session, prompt, maxTokens);

                    await Console.Out.WriteLineAsync();
                    await Console.Out.WriteLineAsync($"Prompt '{prompt}': {history.Length} turns in {sw.Elapsed.TotalSeconds:F2}s  Tokens per second: {session.TokensPerSecond ?? 0:F2}  TTFT: {session.TimeToFirstToken?.ToString("F3") ?? "N/A"}s");
                    await Console.Out.WriteLineAsync();
                }
            }
            await Console.Out.WriteLineAsync($"All Models Executed in: {totalTime.Elapsed.TotalSeconds:F2}s");
            await Console.Out.WriteLineAsync();
        }

        private static async Task<ChatMessage[]> RunPromptAsync(IChatSession session, string prompt, int maxTokens)
        {
            var returnedPrompt = false;
            var tok = 0;
            var cts = new CancellationTokenSource();

            async void Response(ChatStreamEntry text)
            {
                Console.ForegroundColor = text.Status == ChatStatus.Thinking ? ConsoleColor.Gray : ConsoleColor.Blue;
                await Console.Out.WriteAsync(text.Token);
                tok++;
                if (tok > maxTokens) cts.Cancel();
            }

            async Task<ChatMessage> Prompt()
            {
                if (!returnedPrompt && !cts.IsCancellationRequested)
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
                cts.Cancel();
                return new ChatMessage { Content = "exit", Role = ChatRole.User };
            }

            return await session.StartChatAsync(Prompt, Response, cts.Token);
        }
    }
}