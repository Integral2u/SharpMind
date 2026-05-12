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
        private static readonly string ModelName = "qwen2-0_5b-instruct-fp16";
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
                Console.Out.WriteLine($"Tokenizer dat not found");
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
                MaxTokens = 512,
                Temperature = 0.7f,
                TopK = 40,
                TopP = 0.9f,
            };
            if (!string.IsNullOrEmpty(systemPrompt)) session.AddMessage(ChatRole.System, systemPrompt);
            Console.Out.WriteLine("\nChat ready! Say hello.\n");
            var history = await session.StartChatAsync(cancellationTokenSource.Token, Prompt, Response);

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
