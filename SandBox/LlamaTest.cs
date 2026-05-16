using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.Intrinsics.X86;

namespace SandBox
{
    public static class LlamaTest
    {
        private static void SetupNative()
        {
            NativeLibraryConfig.All.WithLogCallback(delegate (LLamaLogLevel level, string message) { });
            NativeLibraryConfig.All.WithAvx(Avx512F.IsSupported ? AvxLevel.Avx512 : Avx2.IsSupported ? AvxLevel.Avx2 : Avx.IsSupported ? AvxLevel.Avx : AvxLevel.None);
        }

        /// <summary>
        /// Verifies SharpMind's prompt formatting produces the correct output
        /// for each model by comparing against the known-good manual format.
        /// </summary>
        public static void VerifyPromptFormatting()
        {
            string assets = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";

            // ── DeepSeek (tests DeepSeekFormatter) ──
            Console.WriteLine("═══ DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M ═══");
            var (dsMeta, dsTok) = LoadMetaAndTokenizer(Path.Combine(assets, "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf"));
            var dsFormatter = ChatPromptFormatterFactory.Create(dsMeta.GetChatTemplate());
            bool addBos = dsMeta.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;

            var dsHistory = new List<ChatMessage>
            {
                new() { Role = ChatRole.System, Content = "You are a helpful assistant." },
                new() { Role = ChatRole.User, Content = "hello" }
            };
            string dsFormatted = dsFormatter.Format(dsHistory, dsTok, addBos);
            Console.WriteLine("Prompt text:");
            Console.WriteLine(dsFormatted);
            Console.WriteLine("Tokens:");
            var dsTokens = dsTok.Encode(dsFormatted, addBos: false);
            Console.WriteLine(string.Join(", ", dsTokens));
            int expectedFirst = dsTok.BosId;
            Console.WriteLine($"First token = {dsTokens[0]} (expected BOS={expectedFirst}): {(dsTokens[0] == expectedFirst ? "✓" : "✗")}");
            Console.WriteLine();

            // ── Llama 3.2 (tests ChatMLFormatter) ──
            Console.WriteLine("═══ Llama-3.2-1B-Instruct-Q8_0 ═══");
            var (llamaMeta, llamaTok) = LoadMetaAndTokenizer(Path.Combine(assets, "llama-3.2-1b-instruct-q8_0.gguf"));
            var llamaFormatter = ChatPromptFormatterFactory.Create(llamaMeta.GetChatTemplate());
            addBos = llamaMeta.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;

            var llamaHistory = new List<ChatMessage>
            {
                new() { Role = ChatRole.System, Content = "You are a helpful assistant." },
                new() { Role = ChatRole.User, Content = "hello" }
            };
            string llamaFormatted = llamaFormatter.Format(llamaHistory, llamaTok, addBos);
            Console.WriteLine("Prompt text:");
            Console.WriteLine(llamaFormatted);
            Console.WriteLine("Tokens:");
            var llamaTokens = llamaTok.Encode(llamaFormatted, addBos: false);
            Console.WriteLine(string.Join(", ", llamaTokens));
            Console.WriteLine();

            // ── Legacy fallback (SimpleFormatter) ──
            Console.WriteLine("═══ No template (SimpleFormatter) ═══");
            var simpleFormatter = new SimpleFormatter();
            string simpleFormatted = simpleFormatter.Format(llamaHistory, llamaTok, addBos: true);
            Console.WriteLine("Prompt text:");
            Console.WriteLine(simpleFormatted);
            Console.WriteLine("Tokens:");
            var simpleTokens = llamaTok.Encode(simpleFormatted, addBos: false);
            Console.WriteLine(string.Join(", ", simpleTokens));
            Console.WriteLine();
        }

        /// <summary>
        /// Runs LLamaSharp inference with a manually-formatted prompt
        /// as the ground-truth reference. Use the prompt text from
        /// VerifyPromptFormatting() to ensure identical inputs.
        /// </summary>
        public static async Task RunLlamaReference()
        {
            SetupNative();
            CancellationTokenSource cts = new();
            string modelPath = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets\DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf";

            var parameters = new ModelParams(modelPath)
            {
                ContextSize = 1024,
                GpuLayerCount = 0
            };

            using var weights = LLamaWeights.LoadFromFile(parameters);
            using var context = weights.CreateContext(parameters);

            var inferenceParams = new InferenceParams()
            {
                MaxTokens = 256,
                AntiPrompts = ["User:", "<｜User｜>"],
                SamplingPipeline = new DefaultSamplingPipeline()
                {
                    Temperature = 0.7f,
                    TopP = 0.9f,
                    TopK = 35,
                }
            };

            // Use the same prompt format that DeepSeekFormatter produces
            var (meta, tok) = LoadMetaAndTokenizer(modelPath);
            var formatter = ChatPromptFormatterFactory.Create(meta.GetChatTemplate());
            bool addBos = meta.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
            var history = new List<ChatMessage>
            {
                new() { Role = ChatRole.User, Content = "hello" }
            };
            string prompt = formatter.Format(history, tok, addBos);

            Console.Out.Write("User: hello\nAssistant: ");
            var executor = new InteractiveExecutor(context);
            await foreach (var text in executor.InferAsync(prompt, inferenceParams, cts.Token))
                Console.Write(text);
            Console.WriteLine();
        }

        // ── Helpers ──

        private static (GgufMeta, Tokenizer) LoadMetaAndTokenizer(string ggufPath)
        {
            var meta = GgufLoader.LoadMeta(ggufPath);
            var tok = GgufLoader.LoadTokenizerFromMeta(meta)
                ?? throw new InvalidOperationException("No tokenizer in GGUF");
            return (meta, tok);
        }
    }
}
