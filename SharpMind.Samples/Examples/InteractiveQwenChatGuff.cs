using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Samples.Examples
{
    public static class InteractiveQwenChatGuff
    {
        private static readonly string ModelName = "qwen2-0_5b-instruct-fp16";
        private static readonly string ModelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        public static async Task RunAsync()
        {
            CancellationTokenSource cancellationTokenSource = new CancellationTokenSource();
            var ggufPath = Path.Combine(ModelPath, $"{ModelName}.gguf");
            var tokenizerPath = Path.Combine(ModelPath, $"{ModelName}.json");
            if (!File.Exists(ggufPath))
            {
                Console.Out.WriteLine($"GGUF not found: {ggufPath}");
                return;
            }
            if (!File.Exists(tokenizerPath))
            {
                Console.Out.WriteLine($"Tokenizer not found: {tokenizerPath}");
                return;
            }
            var sharpConfig = SharpMindConfig.Qwen with { Hardware = DetectBestHardware() };

            Console.Out.WriteLine("Loading tokenizer...");
            var tokenizer = Tokenizer.FromQwen(tokenizerPath);

            Console.Out.WriteLine("Loading Model Config...");
            var meta = GgufLoader.LoadMeta(ggufPath);
            var modelConfig = GgufLoader.LoadConfig(meta);
            if (modelConfig == null)
            {
                Console.Out.WriteLine("Failed to load ModelConfig");
                return;
            }
            GC.Collect(); GC.WaitForPendingFinalizers();
            Console.Out.WriteLine("\nBuilding model...");
            var model = ModelFactory.Create(modelConfig, sharpConfig);
            GC.Collect(); GC.WaitForPendingFinalizers();
            Console.Out.WriteLine("Loading weights...");
            GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
            
            Console.Out.WriteLine("Creating inference ops...");
            var inferOps = InferenceOpsFactory.Create(sharpConfig, InferenceConfig.Default);
            string systemPrompt = $"{PromptHelpers.DefaultSystemPrompt}\n\n{PromptHelpers.DefaultAgentPrompt}".Trim();

            await using var session = new ChatSession(model, tokenizer, inferOps)
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
