using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Samples.Examples
{
    public static class InteractiveChat
    {
        // Set to true for one-shot test, false for interactive
        private const bool TestMode = false;
        private const string TestPrompt = "hello";

        private static readonly string ModelName = "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M";
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync()
        {
            CancellationTokenSource cancellationTokenSource = new();
            var ggufPath = Path.Combine(ModelPath, $"{ModelName}.gguf");
            var tokenizerPath = Path.Combine(ModelPath, $"{ModelName}.json");
            if (!File.Exists(ggufPath))
            {
                Console.Out.WriteLine($"GGUF not found: {ggufPath}");
                return;
            }
            Console.Out.WriteLine("Loading model detail...");
            GgufLoader.Load(ggufPath, null, out GgufMeta meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
            if (tokenizer == null && File.Exists(tokenizerPath)) tokenizer = Tokenizer.FromQwen(tokenizerPath);
            if (tokenizer == null)
            {
                Console.Out.WriteLine($"Tokenizer not found");
                return;
            }
            var sharpConfig = SharpMindConfig.Qwen with { Hardware = DetectBestHardware() };

            GC.Collect(); GC.WaitForPendingFinalizers();
            Console.Out.WriteLine("Building model...");
            var model = ModelFactory.Create(modelConfig, sharpConfig);
            GC.Collect(); GC.WaitForPendingFinalizers();
            Console.Out.WriteLine("Loading weights...");
            GgufLoader.LoadWeightsToModel(ggufPath, meta, model);

            Console.Out.WriteLine("Creating inference ops...");
            var inferOps = InferenceOpsFactory.Create(sharpConfig, InferenceConfig.Default);
            string systemPrompt = $"{PromptHelpers.DefaultSystemPrompt}\n\n{PromptHelpers.DefaultAgentPrompt}".Trim();

            await using var session = new ChatSession(model, tokenizer, inferOps, meta)
            {
                MaxTokens = TestMode ? 50 : 256,
                Temperature = TestMode ? 0.0f : 0.7f,
                TopK = TestMode ? 1 : 35,
                TopP = TestMode ? 0.0f : 0.9f,
                RepetitionPenalty = 1.05f,
                RepetitionWindow = 64,
            };
            if (!string.IsNullOrEmpty(systemPrompt)) session.AddMessage(ChatRole.System, systemPrompt);

            if (TestMode)
            {
                Console.Out.WriteLine($"\nTest prompt: \"{TestPrompt}\"");
                Console.Out.Write("Response: ");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                await foreach (var entry in session.GetResponseStreamAsync(TestPrompt, cancellationTokenSource.Token))
                {
                    if (entry.TextDelta is { Length: > 0 } delta)
                    {
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        Console.Write(delta);
                        Console.ResetColor();
                    }
                }
                sw.Stop();
                Console.Out.WriteLine($"\n--- Completed in {sw.Elapsed.TotalSeconds:F1}s ---");
            }
            else
            {
                Console.Out.WriteLine("\nChat ready! Say hello.\n");
                var history = await session.StartChatAsync(cancellationTokenSource.Token, Prompt, Response);
            }

            void Response(string text)
            {
                Console.ForegroundColor = ConsoleColor.Blue;
                Console.Write(text);
            }

            string Prompt()
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                string userInput = Console.ReadLine() ?? "";
                if (userInput == "exit") cancellationTokenSource.Cancel();
                return userInput;
            }
        }
        private static HardwareTier DetectBestHardware()
        {
            if (Avx2.IsSupported) return HardwareTier.AVX2;
            if (Fma.IsSupported) return HardwareTier.FMA;
            return HardwareTier.Scalar;
        }

    }
}
