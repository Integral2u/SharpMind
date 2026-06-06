using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.Intrinsics.X86;

namespace SharpMind.Samples.Examples
{
    public static class InteractiveChatGuff
    {
        private static readonly string ModelName = "qwen2-0_5b-instruct-q4_k_m";//"SmolLM-135M.Q4_K_M";//"TinyLlama-1.1B-Chat-v1.0.Q4_K_M";// "qwen2-0_5b-instruct-q4_k_m";// SmolLM-135M.Q4_K_M"; //
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
            if (tokenizer == null)
            {
                Console.Out.WriteLine($"Tokenizer dat not found");
                return;
            }
            var sharpConfig = modelConfig.ForModel();

            GC.Collect(); GC.WaitForPendingFinalizers();
            Console.Out.WriteLine("Building model...");
            var model = ModelFactory.Create(modelConfig, sharpConfig);
            GC.Collect(); GC.WaitForPendingFinalizers();
            Console.Out.WriteLine("Loading weights...");
            GgufLoader.LoadWeightsToModel(ggufPath, meta, model);

            await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
            {
                MaxTokens = 512,
                Temperature = 0.7f,
                TopK = 40,
                TopP = 0.9f,
            };
            
            Console.Out.WriteLine("\nChat ready! Say hello.\n");
            var history = await session.StartChatAsync(Prompt, Response, cancellationTokenSource.Token);

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

    }
}
