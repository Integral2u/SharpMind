using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Diagnostics;
using System.Text;

namespace SandBox;

public static class KVCacheBenchmark
{
    private static readonly string Assets = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";

    public static async Task RunAsync()
    {
        var ggufPath = Path.Combine(Assets, "qwen2-0_5b-instruct-q8_0.gguf");
        if (!File.Exists(ggufPath))
        {
            await Console.Error.WriteLineAsync("Model file not found.");
            return;
        }

        GgufLoader.Load(ggufPath, null, out ModelMetaData meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
        if (tokenizer == null)
        {
            await Console.Error.WriteLineAsync("No tokenizer data.");
            return;
        }

        var sharpConfig = modelConfig.ForModel();
        GC.Collect(); GC.WaitForPendingFinalizers();
        var sw = Stopwatch.StartNew();
        var weights = GgufLoader.LoadWeightsToTransformerWeights(ggufPath, modelConfig);
        await Console.Out.WriteLineAsync($"Model load: {sw.Elapsed.TotalSeconds:F2}s\n");

        int[] contextSizes = [512, 1024, 2048];
        int genTokens = 64;

        var cacheTypes = new (string Name, Type Builder)[]
        {
            ("Float", typeof(KVCacherBuilder)),
            ("Q8_0",  typeof(QuantizedKVCacherBuilder)),
        };

        // Build prompts at each context size
        var prompts = new string[contextSizes.Length];
        var actualCtxs = new int[contextSizes.Length];
        for (int i = 0; i < contextSizes.Length; i++)
        {
            var sb = new StringBuilder();
            while (true)
            {
                sb.Append("The quick brown fox jumps over the lazy dog. ");
                int[] toks = tokenizer.Encode(sb.ToString());
                if (toks.Length >= contextSizes[i])
                {
                    prompts[i] = sb.ToString();
                    actualCtxs[i] = toks.Length;
                    break;
                }
            }
        }

        await Console.Out.WriteLineAsync($"{"Ctx",-6} {"Cache",-7} {"Gen",-5} {"t/s",-9} {"TTFT",-9}");
        await Console.Out.WriteLineAsync(new string('-', 42));

        for (int ctxIdx = 0; ctxIdx < contextSizes.Length; ctxIdx++)
        {
            string prompt = prompts[ctxIdx];
            int actualCtx = actualCtxs[ctxIdx];

            for (int c = 0; c < cacheTypes.Length; c++)
            {
                var (name, cacheType) = cacheTypes[c];

                GC.Collect(); GC.WaitForPendingFinalizers();
                var model = ModelFactory.CreateSession(weights, sharpConfig);

                await using var session = ChatSessionFactory.CreateChatSession(
                    typeof(StandardGeneratorBuilder<>), cacheType, model, tokenizer, meta);
                session.MaxTokens = genTokens;
                session.Temperature = 0.0f;
                session.TopK = 1;
                session.StopTokenIds = []; // don't stop on EOS

                int tokCount = 0;
                bool returnedPrompt = false;
                var cts = new CancellationTokenSource();

                async Task<ChatMessage> Prompt()
                {
                    if (!returnedPrompt)
                    {
                        returnedPrompt = true;
                        return new ChatMessage { Content = prompt, Role = ChatRole.User };
                    }
                    cts.Cancel();
                    return new ChatMessage { Content = "exit", Role = ChatRole.User };
                }

                void Response(ChatStreamEntry entry)
                {
                    if (entry.Token != null) tokCount++;
                }

                await session.StartChatAsync(Prompt, Response, cts.Token);

                double tps = session.TokensPerSecond ?? 0;
                float ttft = session.TimeToFirstToken ?? 0;

                await Console.Out.WriteLineAsync(
                    $"{actualCtx,-5} {name,-7} {tokCount,-5} {tps,-9:F3} {ttft,-9:F3}s");
            }
            await Console.Out.WriteLineAsync();
        }

        await Console.Out.WriteLineAsync("Done!");
    }
}
