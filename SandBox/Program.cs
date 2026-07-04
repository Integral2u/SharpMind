using SharpMind;
using SharpMind.Model.Format;
using SharpMind.Model.Config;
using SharpMind.Model;
using SharpMind.Model.Layers;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Tokenization;

await SharpMind.Samples.Examples.KnownFailingModels.RunAsync("Hello");
Console.ReadLine();
return;

string modelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\Qwen3-0.6B-Q4_K_M.gguf";
if (!File.Exists(modelPath))
{
    Console.Error.WriteLine($"Q4_K_M not found, trying Q4_0");
    modelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\Qwen3-0.6B-Q4_0.gguf";
}
if (!File.Exists(modelPath)) { Console.Error.WriteLine("Model not found"); return; }

GgufLoader.Load(modelPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
if (tokenizer == null) { Console.Error.WriteLine("No tokenizer"); return; }

var sharpConfig = modelConfig.ForModel(HardwareTier.Auto);
GC.Collect(); GC.WaitForPendingFinalizers();

using var weights = GgufLoader.LoadWeightsToTransformerWeights(modelPath, modelConfig);
using var model = ModelFactory.CreateSession(weights, sharpConfig);

await using var session = new ChatSession<StandardGeneratorBuilder<KVCacherBuilder>, KVCacherBuilder>(model, tokenizer, meta)
{
    MaxTokens = 30,
    Temperature = 0.0f,
    TopK = 1,
};

bool returnedPrompt = false;
CancellationTokenSource cts = new();

await session.StartChatAsync(
    () =>
    {
        if (!returnedPrompt)
        {
            returnedPrompt = true;
            Console.Error.WriteLine("Prompt: Hello");
            return Task.FromResult(new ChatMessage { Content = "Hello", Role = ChatRole.User });
        }
        cts.Cancel();
        return Task.FromResult<ChatMessage?>(null);
    },
    entry =>
    {
        Console.Out.Write(entry.Token);
        Console.Out.Flush();
    },
    cts.Token);

Console.Error.WriteLine($"\n\nTokens/sec: {session.TokensPerSecond ?? 0:F2}");
