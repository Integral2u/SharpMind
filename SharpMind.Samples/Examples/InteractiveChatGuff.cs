using SharpMind.Core.Quantization;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;

namespace SharpMind.Samples.Examples
{
    public static class InteractiveChatGuff
    {
        public static async Task RunAsync(string ModelPath, string ModelName, LoadMode loadMode = LoadMode.Full, IChatPromptFormatter? formatter = null)
        {
            CancellationTokenSource cancellationTokenSource = new();
            string modelPath = string.Empty;
            ModelFormat? fmt = null;
            foreach (var mFmt in Enum.GetValues<ModelFormat>())
            {
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
                await Console.Out.WriteLineAsync($"Tokenizer dat not found");
                return;
            }
            var sharpConfig = modelConfig.ForModel();
            GC.Collect(); GC.WaitForPendingFinalizers();
            var sw = Stopwatch.StartNew();
            var qOps = QuantizationFactory.Create(sharpConfig.ResolvedHardware);
            using var weights = ModelFactory.CreateWeights(modelConfig, sharpConfig, qOps, modelPath, loadMode);
            weights.InitializeWeights();

            await Console.Out.WriteLineAsync($"ModelFactory.Create + InitializeWeights executed in: {sw.Elapsed.TotalSeconds:F2}s");

            GC.Collect(); GC.WaitForPendingFinalizers();
            sw.Restart();
            using var model = ModelFactory.CreateTransformer(weights, sharpConfig);
            await Console.Out.WriteLineAsync($"ModelFactory.CreateTransformer executed in: {sw.Elapsed.TotalSeconds:F2}s");

            sw.Reset();

            await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta,null,null,null,null,null,null,formatter)
            {
                MaxTokens = 512,
                Temperature = 0.7f,
                TopK = 40,
                TopP = 0.9f,
            };
            session.InitializeChat();
            await Console.Out.WriteLineAsync($"ChatSession executed in: {sw.Elapsed.TotalSeconds:F2}s");
            sw.Stop();
            await Console.Out.WriteLineAsync("\nChat ready! Say hello.\n");
            var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);

            async void Response(ChatStreamEntry text)
            {
                Console.ForegroundColor = text.Status == ChatStatus.Thinking ? ConsoleColor.Gray : ConsoleColor.Blue;
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
