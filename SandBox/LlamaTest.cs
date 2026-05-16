using LLama;
using LLama.Common;
using LLama.Native;
using LLama.Sampling;
using SharpMind;
using SharpMind.Core.Tensors;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Inference.Chat.PromptFormatters;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using System.Runtime.Intrinsics.X86;

namespace SandBox
{
    public static class LlamaTest
    {
        private static readonly string Assets = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";

        private static void SetupNative()
        {
            NativeLibraryConfig.All.WithLogCallback(delegate (LLamaLogLevel level, string message) { });
            NativeLibraryConfig.All.WithAvx(Avx512F.IsSupported ? AvxLevel.Avx512 : Avx2.IsSupported ? AvxLevel.Avx2 : Avx.IsSupported ? AvxLevel.Avx : AvxLevel.None);
        }

        /// <summary>Verifies SharpMind's prompt formatting for each model.</summary>
        public static void VerifyPromptFormatting()
        {
            // DeepSeek
            Console.WriteLine("═══ DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M ═══");
            var (dsMeta, dsTok) = LoadMetaAndTokenizer(Path.Combine(Assets, "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf"));
            VerifyFormatter(dsMeta, dsTok, "DeepSeek");
            Console.WriteLine();

            // Llama 3.2
            Console.WriteLine("═══ Llama-3.2-1B-Instruct-Q8_0 ═══");
            var (llamaMeta, llamaTok) = LoadMetaAndTokenizer(Path.Combine(Assets, "llama-3.2-1b-instruct-q8_0.gguf"));
            VerifyFormatter(llamaMeta, llamaTok, "Llama 3.2");
            Console.WriteLine();

            // Simple fallback
            Console.WriteLine("═══ Simple fallback ═══");
            var simpleFormatter = new SimpleFormatter();
            bool addBos = llamaMeta.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
            var history = MakeHistory();
            string simpleFormatted = simpleFormatter.Format(history, llamaTok, addBos);
            Console.WriteLine(simpleFormatted);
            Console.WriteLine();
        }

        private static void VerifyFormatter(GgufMeta meta, Tokenizer tok, string label)
        {
            var formatter = ChatPromptFormatterFactory.Create(meta.GetChatTemplate());
            bool addBos = meta.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
            string formatted = formatter.Format(MakeHistory(), tok, addBos);
            var encoded = tok.Encode(formatted, addBos: false);
            Console.WriteLine(formatted);
            Console.WriteLine($"Tokens ({encoded.Length}): {string.Join(", ", encoded)}");
        }

        private static List<ChatMessage> MakeHistory() => [
            new() { Role = ChatRole.System, Content = "You are a helpful assistant." },
            new() { Role = ChatRole.User, Content = "hello" }
        ];

        /// <summary>Runs LLamaSharp reference inference with DeepSeek-formatted prompt.</summary>
        public static async Task RunLlamaReference()
        {
            SetupNative();
            CancellationTokenSource cts = new();
            string modelPath = Path.Combine(Assets, "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf");

            var parameters = new ModelParams(modelPath) { ContextSize = 1024, GpuLayerCount = 0 };
            using var weights = LLamaWeights.LoadFromFile(parameters);
            using var context = weights.CreateContext(parameters);

            string prompt = GetDeepSeekPrompt(modelPath);
            Console.Out.Write("User: hello\nAssistant: ");
            var executor = new InteractiveExecutor(context);
            await foreach (var text in executor.InferAsync(prompt, new InferenceParams
            {
                MaxTokens = 256,
                AntiPrompts = ["User:", "<｜User｜>"],
                SamplingPipeline = new DefaultSamplingPipeline { Temperature = 0.7f, TopP = 0.9f, TopK = 35 }
            }, cts.Token))
                Console.Write(text);
            Console.WriteLine();
        }

        /// <summary>
        /// Compares logits between SharpMind and LLamaSharp on the same prompt.
        /// This is the key test to determine if SharpMind's forward pass is correct.
        /// </summary>
        public static async Task CompareLogits()
        {
            Console.WriteLine("═══ Logit Comparison: SharpMind vs LLamaSharp ═══");
            string modelPath = Path.Combine(Assets, "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf");

            // 1. Get the formatted prompt (DeepSeek format)
            string prompt = GetDeepSeekPrompt(modelPath);
            Console.WriteLine($"Prompt: {prompt}");

            // 2. Tokenize with SharpMind tokenizer
            var (meta, sharpTok) = LoadMetaAndTokenizer(modelPath);
            int[] sharpTokens = sharpTok.Encode(prompt, addBos: false);
            Console.WriteLine($"SharpMind tokens ({sharpTokens.Length}): {string.Join(", ", sharpTokens)}");

            // 3. Tokenize with LLamaSharp tokenizer
            SetupNative();
            var llmParams = new ModelParams(modelPath) { ContextSize = 128, GpuLayerCount = 0 };
            using var weights = LLamaWeights.LoadFromFile(llmParams);
            using var context = weights.CreateContext(llmParams);
            var llmTokens = context.Tokenize(prompt, addBos: false, special: true);
            Console.WriteLine($"LLamaSharp tokens ({llmTokens.Length}): {string.Join(", ", llmTokens)}");

            // 4. Compare token IDs
            if (sharpTokens.Length != llmTokens.Length)
            {
                Console.WriteLine($"✗ Token count mismatch: SharpMind={sharpTokens.Length}, LLamaSharp={llmTokens.Length}");
                return;
            }
            bool tokensMatch = true;
            for (int i = 0; i < sharpTokens.Length; i++)
            {
                if ((int)llmTokens[i] != sharpTokens[i])
                {
                    Console.WriteLine($"✗ Token mismatch at position {i}: SharpMind={sharpTokens[i]}, LLamaSharp={llmTokens[i]}");
                    tokensMatch = false;
                }
            }
            if (tokensMatch) Console.WriteLine("✓ Tokens match perfectly!");

            // 5. Run prefill in LLamaSharp and extract last-token logits
            Console.WriteLine("\nRunning LLamaSharp prefill...");
            var batch = new LLamaBatch();
            for (int i = 0; i < llmTokens.Length; i++)
                batch.Add(llmTokens[i], i, LLamaSeqId.Zero, logits: i == llmTokens.Length - 1);
            context.NativeHandle.Decode(batch);
            Span<float> llmLogits = context.NativeHandle.GetLogitsIth(llmTokens.Length - 1);
            var llmTop10 = GetTopK(llmLogits, 10);
            Console.WriteLine("LLamaSharp top-10 logits:");
            foreach (var (id, val) in llmTop10)
                Console.WriteLine($"  [{id}] {sharpTok.IdToToken(id)} = {val:F4}");

            // 6. Load SharpMind model and run prefill
            Console.WriteLine("\nRunning SharpMind prefill...");
            await Task.CompletedTask;
            var smLogits = RunSharpMindInference(modelPath, sharpTokens);
            if (smLogits != null)
            {
                var smTop10 = GetTopK(smLogits.AsSpan(0, Math.Min(smLogits.Length, 151936)), 10);
                Console.WriteLine("SharpMind top-10 logits:");
                foreach (var (id, val) in smTop10)
                    Console.WriteLine($"  [{id}] {sharpTok.IdToToken(id)} = {val:F4}");

                // 7. Compare top-K overlap
                var llmSet = new HashSet<int>(llmTop10.Select(x => x.id));
                var smSet = new HashSet<int>(smTop10.Select(x => x.id));
                int common = llmSet.Intersect(smSet).Count();
                Console.WriteLine($"\nTop-10 overlap: {common}/10");

                var llmTop1 = llmTop10[0];
                var smTop1 = smTop10[0];
                Console.WriteLine($"LLamaSharp top-1: [{llmTop1.id}] {sharpTok.IdToToken(llmTop1.id)} = {llmTop1.val:F4}");
                Console.WriteLine($"SharpMind top-1:  [{smTop1.id}] {sharpTok.IdToToken(smTop1.id)} = {smTop1.val:F4}");
                Console.WriteLine(llmTop1.id == smTop1.id ? "✓ Top-1 matches!" : "✗ Top-1 differs!");
            }

            Console.WriteLine();
        }

        /// <summary>
        /// Reads every tensor from the GGUF file and prints per-tensor stats
        /// (min, max, mean, std, NaN/Inf count). Flags tensors that are all-ones
        /// (norm guard triggered), all-zeros (bias guard / read error), or
        /// contain NaN/Inf values. Shows GGUF dtype to identify broken quantizers.
        /// </summary>
        public static void WeightValidation()
        {
            string modelPath = Path.Combine(Assets, "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf");
            Console.WriteLine("═══ Weight Validation: DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M ═══");

            var meta = GgufLoader.LoadMeta(modelPath);
            var weights = GgufLoader.LoadWeights(modelPath);
            var tensorMap = meta.Tensors.ToDictionary(t => t.Name);

            Console.WriteLine($"Total tensors: {weights.Count}\n");

            int f32 = 0, q3k = 0, q4k = 0, q5k = 0, q6k = 0, broken = 0;

            foreach (var kv in weights.OrderBy(kv => kv.Key))
            {
                var name = kv.Key;
                var tensor = kv.Value;
                var data = tensor.Data;
                int n = data.Length;

                var dtype = tensorMap.TryGetValue(name, out var info) ? info.Dtype : GgufDtype.F32;
                string dtypeStr = dtype switch
                {
                    GgufDtype.F32 => "F32",
                    GgufDtype.F16 => "F16",
                    GgufDtype.Q4_0 => "Q4_0",
                    GgufDtype.Q4_K => "Q4_K",
                    GgufDtype.Q5_0 => "Q5_0",
                    GgufDtype.Q5_K => "Q5_K",
                    GgufDtype.Q6_K => "Q6_K",
                    GgufDtype.Q8_0 => "Q8_0",
                    GgufDtype.Q3_K => "Q3_K",
                    GgufDtype.Q2_K => "Q2_K",
                    _ => $"{(uint)dtype}"
                };

                float min = float.MaxValue, max = float.MinValue;
                double sum = 0, sumSq = 0;
                int nanCount = 0, infCount = 0;
                for (int i = 0; i < n; i++)
                {
                    float v = data[i];
                    if (float.IsNaN(v)) { nanCount++; continue; }
                    if (float.IsInfinity(v)) { infCount++; continue; }
                    if (v < min) min = v;
                    if (v > max) max = v;
                    sum += v;
                    sumSq += (double)v * v;
                }
                int valid = n - nanCount - infCount;
                double mean = valid > 0 ? sum / valid : 0;
                double variance = valid > 0 ? (sumSq / valid) - (mean * mean) : 0;
                double std = variance > 0 ? Math.Sqrt(variance) : 0;

                bool allOnes = valid > 0 && Math.Abs(max - 1f) < 1e-4 && Math.Abs(min - 1f) < 1e-4;
                bool allZeros = valid > 0 && max == 0 && min == 0;
                bool hasNaN = nanCount > 0 || infCount > 0;

                if (hasNaN) broken++;

                string shape = string.Join("×", tensor.Shape.Dims.ToArray());
                string label = name.Contains("norm") || name.Contains("output_norm") ? "NORM" :
                               name.Contains("bias") ? "BIAS" : "";
                string flag = hasNaN ? $" ✗ NaN×{nanCount}" : "";

                if (hasNaN || allOnes || allZeros || (dtype == GgufDtype.F32 && valid > 0 && max > 100))
                    Console.WriteLine($"{name,-50} {dtypeStr,-5} {shape,-14} {min,10:G4} {max,10:G4} {mean,10:G4} {std,10:G4} {label}{flag}");
            }

            Console.WriteLine($"\nBroken (has NaN/Inf): {broken}/{weights.Count}");
            Console.WriteLine($"Breakdown by dtype: F32={f32} Q3_K={q3k} Q4_K={q4k} Q5_K={q5k} Q6_K={q6k}");

            foreach (var kv in weights)
                kv.Value.Dispose();
        }

        // ── Helpers ──

        private static (GgufMeta, Tokenizer) LoadMetaAndTokenizer(string ggufPath)
        {
            var meta = GgufLoader.LoadMeta(ggufPath);
            var tok = GgufLoader.LoadTokenizerFromMeta(meta)
                ?? throw new InvalidOperationException("No tokenizer in GGUF");
            return (meta, tok);
        }

        private static string GetDeepSeekPrompt(string modelPath)
        {
            var (meta, tok) = LoadMetaAndTokenizer(modelPath);
            var formatter = ChatPromptFormatterFactory.Create(meta.GetChatTemplate());
            bool addBos = meta.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
            return formatter.Format(
                [new() { Role = ChatRole.User, Content = "hello" }], tok, addBos);
        }

        private static float[]? RunSharpMindInference(string modelPath, int[] promptTokens)
        {
            try
            {
                GgufLoader.Load(modelPath, null, out GgufMeta meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
                if (tokenizer == null) { Console.WriteLine("No tokenizer from GGUF"); return null; }

                var sharpConfig = SharpMind.SharpMindConfig.Qwen with { Hardware = HardwareTier.AVX2 };
                var model = ModelFactory.Create(modelConfig, sharpConfig);
                GgufLoader.LoadWeightsToModel(modelPath, meta, model);
                var inferOps = InferenceOpsFactory.Create(sharpConfig, InferenceConfig.Default);

                var caches = new KVCache[modelConfig.NumLayers];
                for (int i = 0; i < modelConfig.NumLayers; i++)
                    caches[i] = new KVCache(1, modelConfig.NumKvHeads, 128, modelConfig.HeadDim);

                using var input = SharpMind.Core.Tensors.Tensor<int>.From(promptTokens, 1, promptTokens.Length);
                using var logits = model.ForwardLastLogits(input, caches, 0);
                return logits.Data[..Math.Min(logits.Data.Length, 151936)].ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"SharpMind inference failed: {ex.Message}");
                return null;
            }
        }

        private static List<(int id, float val)> GetTopK(ReadOnlySpan<float> logits, int k)
        {
            var indexed = new (int id, float val)[logits.Length];
            for (int i = 0; i < logits.Length; i++)
                indexed[i] = (i, logits[i]);
            Array.Sort(indexed, (a, b) => b.val.CompareTo(a.val));
            return indexed.Take(k).ToList();
        }
    }
}
