using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Samples.Examples
{
    public static class InteractiveChat
    {
        private const string TestPrompt = "hello";

        private static readonly string ModelName = "TinyLlama-1.1B-Chat-v1.0.Q4_K_M";
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
            GgufLoader.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
            if (tokenizer == null && File.Exists(tokenizerPath)) tokenizer = Tokenizer.FromQwen(tokenizerPath);
            if (tokenizer == null)
            {
                Console.Out.WriteLine($"Tokenizer not found");
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
                MaxTokens = 256,
                Temperature = 0.7f,
                TopK = 35,
                TopP = 0.9f,
                RepetitionPenalty = 1.1f,
                RepetitionWindow = 64,
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
