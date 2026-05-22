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

        public static async Task TestChat()
        {
            NativeLibraryConfig.All.WithLogCallback(delegate (LLamaLogLevel level, string message) { });
            NativeLibraryConfig.All.WithAvx(Avx512F.IsSupported ? AvxLevel.Avx512 : Avx2.IsSupported ? AvxLevel.Avx2 : Avx.IsSupported ? AvxLevel.Avx : AvxLevel.None);

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

            string prompt = "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜>\n";

            // Test 1: InteractiveExecutor + manual prompt (WORKS)
            Console.WriteLine("=== InteractiveExecutor + manual prompt ===");
            var executor1 = new InteractiveExecutor(context);
            Console.Out.Write("User: hello\nAssistant: ");
            await foreach (var text in executor1.InferAsync(prompt, inferenceParams, cts.Token))
                Console.Write(text);
            Console.WriteLine("\n");

            // Test 2: ChatSession + ChatHistory (BROKEN for DeepSeek)
            Console.WriteLine("=== ChatSession + ChatHistory ===");
            using var context2 = weights.CreateContext(parameters);
            var executor2 = new InteractiveExecutor(context2);
            var session = new LLama.ChatSession(executor2, new ChatHistory([
                new(AuthorRole.System, "You are a helpful assistant.")
            ]));
            Console.Out.Write("User: hello\nAssistant: ");
            await foreach (var text in session.ChatAsync(
                new ChatHistory.Message(AuthorRole.User, "hello"),
                inferenceParams, cts.Token))
                Console.Write(text);
            Console.WriteLine();

            Console.ReadLine();
        }
        /// <summary>Verifies SharpMind's prompt formatting for each model.</summary>
        public static void VerifyPromptFormatting()
        {
            // DeepSeek
            Console.WriteLine("═══ DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M ═══");
            var (dsMeta, dsTok) = LoadMetaAndTokenizer(Path.Combine(Assets, "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf"));
            VerifyFormatter(dsMeta, dsTok);
            Console.WriteLine();

            // Llama 3.2
            Console.WriteLine("═══ Llama-3.2-1B-Instruct-Q8_0 ═══");
            var (llamaMeta, llamaTok) = LoadMetaAndTokenizer(Path.Combine(Assets, "llama-3.2-1b-instruct-q8_0.gguf"));
            VerifyFormatter(llamaMeta, llamaTok);
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

        private static void VerifyFormatter(GgufMeta meta, Tokenizer tok)
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

            // 0. Debug: check GGUF tensor shapes
            var (dbgMeta, _) = LoadMetaAndTokenizer(modelPath);
            foreach (var t in dbgMeta.Tensors)
                if (t.Name.Contains("token_embd") || t.Name.Contains("output"))
                    Console.WriteLine($"[{t.Name}] shape=[{string.Join(",", t.Shape)}] dtype={t.Dtype} offset={t.Offset}");

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

            // Use known-good reference logits from the earlier short-prompt run
            // instead of calling LLamaSharp which hangs
            int[] checkToks = [71486, 32313, 9707, 13048, 40, 0, 1, 151646];
            Console.WriteLine("\n═══ Using reference logits (LLamaSharp skipped) ═══");
            for (int t = 0; t < llmTokens.Length; t++)
            {
                Console.Write($"Token {t} (id={sharpTokens[t]}):");
                foreach (int cid in checkToks)
                    Console.Write($" [{cid}]={0,8:G4}(ref)");
                Console.WriteLine();
            }
            var llmTop10 = new List<(int id, float val)> {
                (71486, 24.0048f), (32313, 23.1231f), (9707, 19.1231f), (13048, 18.1466f), (40, 16.5876f),
                (18665, 16.5190f), (80022, 16.2155f), (106287, 16.0980f), (108386, 16.0230f), (0, 15.7615f)
            };
            Console.WriteLine("LLamaSharp top-10 logits (from earlier run):");
            foreach (var (id, val) in llmTop10.Take(3))
                Console.WriteLine($"  [{id}] {sharpTok.IdToToken(id)} = {val:F4}");
            Console.WriteLine("  ...");

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

        // Check norm weights specifically
        Console.WriteLine("═══ Norm Weights (ffn_norm) ═══");
        foreach (var kv in weights.OrderBy(kv => kv.Key))
        {
            if (!kv.Key.Contains("ffn_norm")) continue;
            var data = kv.Value.Data;
            float min = float.MaxValue, max = float.MinValue;
            double sum = 0;
            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (v < min) min = v; if (v > max) max = v; sum += v;
            }
            int layer = int.Parse(System.Text.RegularExpressions.Regex.Match(kv.Key, @"\d+").Value);
            if (layer <= 2)
                Console.WriteLine($"{kv.Key,-45} min={min,8:G4} max={max,8:G4} mean={sum/data.Length,8:G4}");
        }

        // Also check attn_norm
        Console.WriteLine("\n═══ Norm Weights (attn_norm) ═══");
        foreach (var kv in weights.OrderBy(kv => kv.Key))
        {
            if (!kv.Key.Contains("attn_norm")) continue;
            var data = kv.Value.Data;
            float min = float.MaxValue, max = float.MinValue;
            double sum = 0;
            for (int i = 0; i < data.Length; i++)
            {
                float v = data[i];
                if (v < min) min = v; if (v > max) max = v; sum += v;
            }
            int layer = int.Parse(System.Text.RegularExpressions.Regex.Match(kv.Key, @"\d+").Value);
            if (layer <= 2)
                Console.WriteLine($"{kv.Key,-45} min={min,8:G4} max={max,8:G4} mean={sum/data.Length,8:G4}");
        }

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

        /// <summary>
        /// Layer-by-layer NaN tracer. Runs the forward pass manually,
        /// checking for NaN after every operation, and reports where
        /// NaN first appears.
        /// </summary>
        public static void TraceForwardPass()
        {
            string modelPath = Path.Combine(Assets, "DeepSeek-R1-Distill-Qwen-1.5B-Q3_K_M.gguf");
            Console.WriteLine("═══ Trace Forward Pass NaN Cascade ═══");

            GgufLoader.Load(modelPath, null, out GgufMeta meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
            if (tokenizer == null) { Console.WriteLine("No tokenizer"); return; }

            Console.WriteLine($"Architecture: {meta.GetString("general.architecture")}");
            Console.WriteLine($"RopeTheta: {modelConfig.RopeTheta}");
            Console.WriteLine($"NormEps: {modelConfig.NormEps}");
            Console.WriteLine($"FfnDim: {modelConfig.FfnDim}");
            Console.WriteLine($"MaxSeqLen: {modelConfig.MaxSeqLen}");

            var sharpConfig = DeriveSharpMindConfig(modelConfig, HardwareTier.AVX2);
            var model = ModelFactory.Create(modelConfig, sharpConfig);
            GgufLoader.LoadWeightsToModel(modelPath, meta, model);

            var prompt = GetDeepSeekPrompt(modelPath);
            int[] tokens = tokenizer.Encode(prompt, addBos: false);
            Console.WriteLine($"Prompt tokens: {string.Join(", ", tokens)}");

            using var input = SharpMind.Core.Tensors.Tensor<int>.From(tokens, 1, tokens.Length);

            // 1. Embedding
            Console.WriteLine("Stage 1: Embedding...");
            var embedded = model.ForwardEmbedding(input);
            var embNaN = CountNaN(embedded.Data);
            Console.WriteLine($"  Embedding NaN: {embNaN}/{embedded.Data.Length} ({100f * embNaN / embedded.Data.Length:F2}%)");
            if (embNaN > 0) { Console.WriteLine("✗ NaN in embedding!"); return; }

            // 2. Layer-by-layer
            var hidden = embedded;
            var caches = new KVCache[modelConfig.NumLayers];
            for (int i = 0; i < modelConfig.NumLayers; i++)
                caches[i] = new KVCache(1, modelConfig.NumKvHeads, 128, modelConfig.HeadDim);

            for (int layer = 0; layer < modelConfig.NumLayers; layer++)
            {
                Console.Write($"Layer {layer}: ");
                var block = model.GetBlock(layer);
                if (block == null) { Console.WriteLine("null block"); return; }

                var ops = model.Ops; // TensorOps

                // Pre-attention norm
                var normed1 = block.Norm1.Forward(hidden);
                int n1 = CountNaN(normed1.Data);
                if (n1 > 0) { Console.WriteLine($"✗ norm1 NaN: {n1}/{normed1.Data.Length}"); return; }

                // ── Trace attention sub-steps ──
                var attn = block.Attention;
                var q = attn.Wq.Forward(normed1, ops);  int qn = CountNaN(q.Data);
                var k = attn.Wk.Forward(normed1, ops);  int kn = CountNaN(k.Data);
                var v = attn.Wv.Forward(normed1, ops);  int vn = CountNaN(v.Data);

                if (qn > 0) { Console.WriteLine($"✗ Q NaN: {qn}"); normed1.Dispose(); q.Dispose(); k.Dispose(); v.Dispose(); return; }
                if (kn > 0) { Console.WriteLine($"✗ K NaN: {kn}"); normed1.Dispose(); q.Dispose(); k.Dispose(); v.Dispose(); return; }
                if (vn > 0) { Console.WriteLine($"✗ V NaN: {vn}"); normed1.Dispose(); q.Dispose(); k.Dispose(); v.Dispose(); return; }

                Console.Write($"(Q={MaxAbs(q.Data):G3} K={MaxAbs(k.Data):G3}");
                Console.Write($" V={MaxAbs(v.Data):G3}) ");
                Console.Write($"attn_in={MaxAbs(normed1.Data):G3} ");

                // Run full attention, then check output
                var attnOut = block.Attention.Forward(normed1, ops, 0, true, caches[layer]);
                q.Dispose(); k.Dispose(); v.Dispose();
                int aNaN = CountNaN(attnOut.Data);
                if (aNaN > 0) { Console.WriteLine($"✗ attention NaN: {aNaN}/{attnOut.Data.Length}"); normed1.Dispose(); attnOut.Dispose(); return; }
                Console.Write($"attnOut={MaxAbs(attnOut.Data):G3} ");

                // Residual
                var h1 = SharpMind.Core.Ops.TensorOps.Add(hidden, attnOut);
                attnOut.Dispose();
                hidden.Dispose();
                normed1.Dispose();

                // Pre-FFN norm
                var normed2 = block.Norm2.Forward(h1);
                int n2 = CountNaN(normed2.Data);
                if (n2 > 0) { Console.WriteLine($"✗ norm2 NaN: {n2}/{normed2.Data.Length}"); return; }

                // FFN
                var ffnOut = block.Ffn.Forward(normed2);
                int fNaN = CountNaN(ffnOut.Data);

                if (layer <= 2)
                {
                    var wg = block.Ffn.WGate; var wu = block.Ffn.WUp;
                    Console.Write($"nin={MaxAbs(normed2.Data):G3} h1max={MaxAbs(h1.Data):G3} ");
                    int hd = modelConfig.HiddenDim;
                    // Find position of max normed value
                    int nTok = 0, nDim = 0; float maxN = 0;
                    for (int t = 0; t < 5; t++)
                        for (int d = 0; d < hd; d++)
                        {
                            float val = Math.Abs(normed2.Data[t * hd + d]);
                            if (val > maxN) { maxN = val; nTok = t; nDim = d; }
                        }
                    // Compute RMS of h1 at that token
                    double hss = 0; var hrow = h1.Data.Slice(nTok * hd, hd);
                    for (int ii = 0; ii < hd; ii++) hss += hrow[ii] * hrow[ii];
                    float hrms = (float)Math.Sqrt(hss / hd + 1e-5f);
                    // weight = normed * h1_rms / h1
                    float h1atN = Math.Abs(h1.Data[nTok * hd + nDim]);
                    float wImplied = maxN * hrms / Math.Max(h1atN, 1e-10f);
                    Console.Write($"maxNin@tok{nTok}d{nDim}: h1={h1atN:G3} h1rms={hrms:G3} w~={wImplied:G3} ");
                    using var gate = wg.Forward(normed2, ops);
                    using var up = wu.Forward(normed2, ops);
                    Console.Write($"gateO={MaxAbs(gate.Data):G3} upO={MaxAbs(up.Data):G3} ");
                }

                normed2.Dispose();
                if (fNaN > 0) { Console.WriteLine($"✗ FFN NaN: {fNaN}/{ffnOut.Data.Length}"); return; }
                Console.Write($"ffnOut={MaxAbs(ffnOut.Data):G3} ");

                // Residual: out = h + ffn
                var output = SharpMind.Core.Ops.TensorOps.Add(h1, ffnOut);
                ffnOut.Dispose();
                h1.Dispose();
                hidden = output;

                var oNaN = CountNaN(hidden.Data);
                if (oNaN > 0) { Console.WriteLine($"✗ output NaN: {oNaN}/{hidden.Data.Length}"); return; }
                Console.WriteLine($"hidden={MaxAbs(hidden.Data):G3}");
            }

            // 3. Final norm
            Console.Write("Final norm: ");
            var finalNormed = model.FinalNorm.Forward(hidden);
            int fnNaN = CountNaN(finalNormed.Data);
            if (fnNaN > 0) { Console.WriteLine($"✗ NaN: {fnNaN}/{finalNormed.Data.Length}"); return; }
            Console.WriteLine("ok");

            // 4. LM head
            Console.Write("LM head: ");
            int lastPos = (tokens.Length - 1) * modelConfig.HiddenDim;
            var lastNormed = new SharpMind.Core.Tensors.Tensor<float>(1, modelConfig.HiddenDim);
            finalNormed.Data.Slice(lastPos, modelConfig.HiddenDim).CopyTo(lastNormed.Data);
            Console.Write($"normed max={MaxAbs(lastNormed.Data):G3} ");

            var projW = model.LmHead ?? model.EmbeddingWeight;
            Console.Write($"projW shape=[{projW.Shape.Dims.ToArray()[0]}×{projW.Shape.Dims.ToArray()[1]}] ");

            // Manual dot product for first 5 logits
            Console.Write("manual:");
            for (int j = 0; j < 5 && j < projW.Shape.Rows; j++)
            {
                double sum = 0;
                for (int k = 0; k < modelConfig.HiddenDim; k++)
                    sum += lastNormed.Data[k] * projW.Data[j * modelConfig.HiddenDim + k];
                Console.Write($" {sum:G4}");
            }

            // Manual dot product for LLamaSharp's top tokens
            int[] checkTokens = [71486, 32313, 9707, 13048, 11578, 40];
            Console.Write(" checkTok:");
            foreach (int t in checkTokens)
            {
                double sum = 0;
                for (int k = 0; k < modelConfig.HiddenDim; k++)
                    sum += lastNormed.Data[k] * projW.Data[t * modelConfig.HiddenDim + k];
                Console.Write($" [{t}]={sum:G4}");
            }

            var logits = model.Ops.MatMulWithBT(lastNormed, projW);
            int lNaN = CountNaN(logits.Data);
            Console.WriteLine(lNaN > 0 ? $" ✗ NaN: {lNaN}/{logits.Data.Length}" : "");

            var top10 = GetTopK(logits.Data, 10);
            Console.WriteLine("Top-10 logits:");
            foreach (var (id, val) in top10)
                Console.WriteLine($"  [{id}] {tokenizer.IdToToken(id)} = {val:F4}");

            finalNormed.Dispose();
            lastNormed.Dispose();
            logits.Dispose();
            hidden.Dispose();
            foreach (var c in caches) c.Dispose();
            model.Dispose();
        }

        private static int CountNaN(ReadOnlySpan<float> data)
        {
            int c = 0;
            for (int i = 0; i < data.Length; i++) if (float.IsNaN(data[i])) c++;
            return c;
        }

        private static float MaxAbs(ReadOnlySpan<float> data)
        {
            float m = 0;
            for (int i = 0; i < data.Length; i++) { float a = Math.Abs(data[i]); if (a > m) m = a; }
            return m;
        }

        private static float MeanAbs(ReadOnlySpan<float> data)
        {
            double sum = 0;
            for (int i = 0; i < data.Length; i++) sum += Math.Abs(data[i]);
            return (float)(sum / data.Length);
        }

        // ── Helpers ──

        /// <summary>
        /// Derives the correct <see cref="SharpMindConfig"/> from the model's
        /// GGUF-loaded <see cref="ModelConfig"/>.  Reads NumKvHeads vs NumHeads
        /// to pick MHA / GQA / MQA so the caller never needs to hardcode an
        /// architecture name.
        /// </summary>
        private static global::SharpMind.SharpMindConfig DeriveSharpMindConfig(ModelConfig config, HardwareTier hw)
            => global::SharpMind.SharpMindConfig.ForModel(config.NumHeads, config.NumKvHeads, hw);

        /// <summary>
        /// Runs per-layer diagnostics on any model.
        /// Loads the model with auto-detected config, runs forward pass for a
        /// test prompt, and dumps hidden-state + logit stats after each layer.
        /// </summary>
        public static void RunDiagnostics(string modelPath)
        {
            Console.WriteLine($"═══ Diagnostics: {Path.GetFileName(modelPath)} ═══");

            GgufLoader.Load(modelPath, null, out GgufMeta meta, out ModelConfig modelConfig, out Tokenizer? tokenizer);
            if (tokenizer == null) { Console.WriteLine("No tokenizer"); return; }

            string arch = meta.GetString("general.architecture") ?? "?";
            string? chatTemplate = meta.GetChatTemplate();
            Console.WriteLine($"Architecture: {arch}");
            Console.WriteLine($"Chat template: {chatTemplate ?? "(none)"}");
            Console.WriteLine($"HiddenDim={modelConfig.HiddenDim} NumLayers={modelConfig.NumLayers} " +
                              $"NumHeads={modelConfig.NumHeads} NumKvHeads={modelConfig.NumKvHeads} " +
                              $"HeadDim={modelConfig.HeadDim} FfnDim={modelConfig.FfnDim} " +
                              $"RopeTheta={modelConfig.RopeTheta} NormEps={modelConfig.NormEps}");
            Console.WriteLine($"VocabSize={modelConfig.VocabSize} MaxSeqLen={modelConfig.MaxSeqLen}");

            var sharpConfig = DeriveSharpMindConfig(modelConfig, HardwareTier.AVX2);
            Console.WriteLine($"SharpMindConfig: Attn={sharpConfig.Attention} Ffn={sharpConfig.Ffn} " +
                              $"Act={sharpConfig.Activation} Norm={sharpConfig.Norm}");

            var model = ModelFactory.Create(modelConfig, sharpConfig);
            GgufLoader.LoadWeightsToModel(modelPath, meta, model);

            // Build prompt using the model's chat template
            var formatter = ChatPromptFormatterFactory.Create(meta.GetChatTemplate());
            bool addBos = meta.GetLong("tokenizer.ggml.add_bos_token", 1) != 0;
            string prompt = formatter.Format(
                [new() { Role = ChatRole.User, Content = "hello" }], tokenizer, addBos);
            Console.WriteLine($"Prompt ({prompt.Length} chars): {prompt.Replace("\n", "\\n")}");

            int[] tokenIds = tokenizer.Encode(prompt, addBos: false);
            Console.WriteLine($"Tokens ({tokenIds.Length}): {string.Join(", ", tokenIds.Take(32))}{(tokenIds.Length > 32 ? "..." : "")}");

            using var input = SharpMind.Core.Tensors.Tensor<int>.From(tokenIds, 1, tokenIds.Length);
            int hiddenDim = modelConfig.HiddenDim;
            int seqLen = tokenIds.Length;

            // Create KV caches
            var caches = new KVCache[modelConfig.NumLayers];
            for (int i = 0; i < modelConfig.NumLayers; i++)
                caches[i] = new KVCache(1, modelConfig.NumKvHeads, 128, modelConfig.HeadDim);

            // Embedding
            Console.WriteLine("\n─── Embedding ───");
            var hidden = model.ForwardEmbedding(input);
            var embStats = TensorStats(hidden.Data, hiddenDim);
            Console.WriteLine($"  hMax={embStats.maxAbs:G5} hMean={embStats.meanAbs:G5} NaN={embStats.nanCnt}");

            var projW = model.LmHead ?? model.EmbeddingWeight;

            // Check embedding-only logits
            int[] checkTokens = [0, 1, 2, tokenizer.BosId, tokenizer.EosId];
            var checkSet = new HashSet<int>(checkTokens);
            // Add a few more likely tokens
            for (int t = 3; t < 10 && t < modelConfig.VocabSize; t++) checkSet.Add(t);
            checkTokens = [.. checkSet];

            ReportLogits("Embedding", hidden, projW, checkTokens, hiddenDim, tokenizer);

            // Per-layer loop
            for (int layer = 0; layer < modelConfig.NumLayers; layer++)
            {
                var block = model.GetBlock(layer);
                if (block == null) { Console.WriteLine($"  Layer {layer}: null block — aborting"); break; }

                var normed1 = block.Norm1.Forward(hidden);
                var attnOut = block.Attention.Forward(normed1, model.Ops, positionOffset: 0, causal: true, cache: caches[layer]);
                normed1.Dispose();
                var h1 = SharpMind.Core.Ops.TensorOps.Add(hidden, attnOut);
                attnOut.Dispose();
                var normed2 = block.Norm2.Forward(h1);
                var ffnOut = block.Ffn.Forward(normed2);
                normed2.Dispose();
                var output = SharpMind.Core.Ops.TensorOps.Add(h1, ffnOut);
                h1.Dispose();
                ffnOut.Dispose();

                if (layer > 0) hidden.Dispose();
                hidden = output;

                var stats = TensorStats(hidden.Data, hiddenDim);
                Console.Write($"  L{layer:D2}: hMax={stats.maxAbs:G5} hMean={stats.meanAbs:G5} NaN={stats.nanCnt}");
                if (stats.nanCnt > 0) { Console.WriteLine(" ✗ NaN detected! Stopping."); break; }
                if (stats.allZero) { Console.WriteLine(" ✗ ALL ZERO! Stopping."); break; }

                ReportLogits($"L{layer:D2}", hidden, projW, checkTokens, hiddenDim, tokenizer);
            }

            // Final norm
            Console.WriteLine("\n─── Final Norm ───");
            var finalNormed = model.FinalNorm.Forward(hidden);
            var (maxAbs, meanAbs, nanCnt, allZero) = TensorStats(finalNormed.Data, hiddenDim);
            Console.WriteLine($"  fnMax={maxAbs:G5} fnMean={meanAbs:G5} NaN={nanCnt}");

            // LM head
            Console.WriteLine("\n─── LM Head ───");
            var lastNormed = finalNormed.Data.Slice((seqLen - 1) * hiddenDim, hiddenDim);
            if (model.LmHead != null)
            {
                Console.Write("  Manual dot (first 5 logits):");
                for (int j = 0; j < 5 && j < projW.Shape.Rows; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < hiddenDim; k++) sum += lastNormed[k] * projW.Data[j * hiddenDim + k];
                    Console.Write($" {sum:G4}");
                }
                Console.WriteLine();
            }

            var logitsFlat = finalNormed.Reshape(seqLen, hiddenDim);
            using var logits = model.Ops.MatMulWithBT(logitsFlat, projW);
            int vocabSize = Math.Min(logits.Shape[1], modelConfig.VocabSize);
            var lastLogits = logits.Data.Slice((seqLen - 1) * vocabSize, vocabSize);
            var top10 = GetTopK(lastLogits, 10);
            Console.WriteLine("  Top-10 logits:");
            foreach (var (id, val) in top10)
                Console.WriteLine($"    [{id}] {tokenizer.IdToToken(id) ?? "?"} = {val:F4}");

            Console.WriteLine($"\n═══ Diagnostics complete ═══");

            finalNormed.Dispose();
            logitsFlat.Dispose();
            hidden.Dispose();
            foreach (var c in caches) c.Dispose();
            model.Dispose();
        }

        private static void ReportLogits(string label, Tensor<float> hidden, Tensor<float> projW,
            int[] checkTokens, int hiddenDim, Tokenizer? tokenizer)
        {
            int seqLen = hidden.Shape[1];
            var lastToken = hidden.Data.Slice((seqLen - 1) * hiddenDim, hiddenDim);
            Console.Write($"  {label} logits:");
            foreach (int t in checkTokens)
            {
                if (t < 0 || t >= projW.Shape.Rows) continue;
                double sum = 0;
                for (int k = 0; k < hiddenDim; k++)
                    sum += lastToken[k] * projW.Data[t * hiddenDim + k];
                var tok = tokenizer?.IdToToken(t) ?? "?";
                string display = tok.Length > 8 ? tok[..8] + "…" : tok;
                Console.Write($" [{t}:{display}]={sum,8:G4}");
            }
            Console.WriteLine();
        }

        private static (float maxAbs, float meanAbs, int nanCnt, bool allZero) TensorStats(
            ReadOnlySpan<float> data, int hiddenDim)
        {
            int seqLen = data.Length / hiddenDim;
            var last = data.Slice((seqLen - 1) * hiddenDim, hiddenDim);
            float maxAbs = 0, meanAbs = 0;
            int nanCnt = 0;
            bool allZero = true;
            for (int i = 0; i < last.Length; i++)
            {
                float v = last[i];
                if (float.IsNaN(v)) { nanCnt++; continue; }
                float a = Math.Abs(v);
                if (a > maxAbs) maxAbs = a;
                meanAbs += a;
                if (a > 1e-10f) allZero = false;
            }
            meanAbs /= last.Length;
            return (maxAbs, meanAbs, nanCnt, allZero);
        }

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

                var sharpConfig = DeriveSharpMindConfig(modelConfig, HardwareTier.Scalar);
                var model = ModelFactory.Create(modelConfig, sharpConfig);
                GgufLoader.LoadWeightsToModel(modelPath, meta, model);
                var inferOps = InferenceOpsFactory.Create(sharpConfig, InferenceConfig.Default);

                // Diagnostic: check LM head
                var lmHead = model.LmHead;
                Console.WriteLine($"LM head loaded: {(lmHead != null ? $"shape=[{lmHead.Shape.Rows},{lmHead.Shape.Cols}]" : "null (using embedding)")}");
                
                // Check token 71486 ("Alright") LM head weight
                int checkToken = 71486;
                if (lmHead != null)
                {
                    var row = lmHead.RowSpan(checkToken);
                    float min = float.MaxValue, max = float.MinValue;
                    for (int i = 0; i < row.Length; i++) { float v = row[i]; if (v < min) min = v; if (v > max) max = v; }
                    double mean = 0; for (int i = 0; i < row.Length; i++) mean += row[i]; mean /= row.Length;
                    Console.WriteLine($"  token {checkToken} weight: min={min:G4} max={max:G4} mean={mean:G4}");
                    Console.Write($"  token {checkToken} first 10: ");
                    for (int k = 0; k < 10; k++) Console.Write($"{row[k]:G6} ");
                }
                else
                {
                    var embedWeight = model.EmbeddingWeight;
                    var row = embedWeight.RowSpan(checkToken);
                    float min = float.MaxValue, max = float.MinValue;
                    for (int i = 0; i < row.Length; i++) { float v = row[i]; if (v < min) min = v; if (v > max) max = v; }
                    Console.WriteLine($"  (embedding) token {checkToken} weight: min={min:G4} max={max:G4}");
                }

                using var input = SharpMind.Core.Tensors.Tensor<int>.From(promptTokens, 1, promptTokens.Length);
                int hiddenDim = modelConfig.HiddenDim;
                int seqLen = promptTokens.Length;
                int[] checkToks = [71486, 32313, 9707, 13048, 40, 0, 1, 151646]; // LLamaSharp top-8 roughly

                // Create KVCaches
                var caches = new KVCache[modelConfig.NumLayers];
                for (int ci = 0; ci < modelConfig.NumLayers; ci++)
                    caches[ci] = new KVCache(1, modelConfig.NumKvHeads, 128, modelConfig.HeadDim);

                // ── Step 1: Embedding diagnostics ──
                using var embedded = model.ForwardEmbedding(input);
                Console.WriteLine($"\n═══ Embedding diagnostics ═══");
                for (int t = 0; t < seqLen; t++)
                {
                    var row = embedded.Data.Slice(t * hiddenDim, hiddenDim);
                    Console.WriteLine($"  Token {t} (id={promptTokens[t]}): max={MaxAbs(row):G4} meanAbs={MeanAbs(row):G4}");
                }
                Console.Write($"  Token 0 first 10 values: ");
                for (int k = 0; k < 10; k++) Console.Write($"{embedded.Data[k]:G4} ");
                Console.WriteLine();



                // Quick sanity: compute emb[T0] @ LMHead to see if top-1 makes sense
                Console.WriteLine("\nEmbedding-only logits (no layers):");
                var projWCheck = model.LmHead ?? model.EmbeddingWeight;
                Console.Write($"  Manual dot for token 0: ");
                var embRow0 = embedded.Data[..hiddenDim];
                double[] checkVals = new double[checkToks.Length];
                for (int ci = 0; ci < checkToks.Length; ci++)
                {
                    double sum = 0;
                    for (int k = 0; k < hiddenDim; k++)
                        sum += embRow0[k] * projWCheck.Data[checkToks[ci] * hiddenDim + k];
                    checkVals[ci] = sum;
                }
                foreach (int ci in checkToks)
                {
                    int idx = Array.IndexOf(checkToks, ci);
                    Console.Write($" [{ci}]={checkVals[idx],8:G4}");
                }
                Console.WriteLine();

                // Compute full embedding-only logits and show top-5
                using var embeddedFlat = embedded.Reshape(seqLen, hiddenDim);
                using var embLogits = model.Ops.MatMulWithBT(embeddedFlat, projWCheck);
                int evs = Math.Min(embLogits.Data.Length / seqLen, 151936);
                var embTop5 = GetTopK(embLogits.Data[..evs], 5);
                Console.WriteLine("  Embedding-only top-5:");
                foreach (var (id, val) in embTop5)
                    Console.WriteLine($"    [{id}] {tokenizer?.IdToToken(id) ?? "?"} = {val:F4}");

                // ── Per-layer diagnostic loop ──
                Console.WriteLine("Running per-layer diagnostics...");
                var hiddenState = embedded; // [1, seqLen, hiddenDim]
                var projW = model.LmHead ?? model.EmbeddingWeight;
                for (int layerIdx = 0; layerIdx < modelConfig.NumLayers; layerIdx++)
                {
                    var block = model.GetBlock(layerIdx);
                    
                    // ── Layer 0 step-by-step diagnostics ──
                    if (layerIdx == 0)
                    {
                        // Norm1 output for token 0
                        using var n1 = block.Norm1.Forward(hiddenState);
                        Console.Write($"  L00 Norm1 out[0][0..20]: ");
                        for (int k = 0; k < 20; k++) Console.Write($"{n1.Data[k]:G6} ");
                        Console.WriteLine();
                        
                        // Manual QKV projections for comparison
                        using var qProj = block.Attention.Wq.Forward(hiddenState, model.Ops);
                        using var kProj = block.Attention.Wk.Forward(hiddenState, model.Ops);
                        using var vProj = block.Attention.Wv.Forward(hiddenState, model.Ops);
                        Console.Write($"  L00 Q[0][0..20]: ");
                        for (int k = 0; k < 20; k++) Console.Write($"{qProj.Data[k]:G6} ");
                        Console.WriteLine();
                        Console.Write($"  L00 Q maxAbs: {MaxAbs(qProj.Data):G6}");
                        Console.Write($"  K maxAbs: {MaxAbs(kProj.Data):G6}");
                        Console.WriteLine($"  V[0][0..10]: {string.Join(" ", vProj.Data[0..10].ToArray().Select(v => v.ToString("G6")))}");
                        
                        // Full attention (includes RoPE internally)
                        var attnOut = block.Attention.Forward(n1, model.Ops, positionOffset: 0, causal: true, cache: caches[layerIdx]);
                        Console.Write($"  L00 attnOut[0][0..10]: ");
                        for (int k = 0; k < 10; k++) Console.Write($"{attnOut.Data[k]:G6} ");
                        Console.WriteLine();
                        n1.Dispose();
                        
                        // Residual
                        var h1 = SharpMind.Core.Ops.TensorOps.Add(hiddenState, attnOut);
                        attnOut.Dispose();
                        
                        // Norm2
                        using var n2 = block.Norm2.Forward(h1);
                        Console.Write($"  L00 Norm2 out[0][0..20]: ");
                        for (int k = 0; k < 20; k++) Console.Write($"{n2.Data[k]:G6} ");
                        Console.WriteLine();
                        
                        // Manual FFN gate/up
                        using var gate = block.Ffn.WGate.Forward(n2, model.Ops);
                        using var up = block.Ffn.WUp.Forward(n2, model.Ops);
                        Console.Write($"  L00 Gate[0][0..10]: ");
                        for (int k = 0; k < 10; k++) Console.Write($"{gate.Data[k]:G6} ");
                        Console.WriteLine();
                        Console.Write($"  L00 Up[0][0..10]: ");
                        for (int k = 0; k < 10; k++) Console.Write($"{up.Data[k]:G6} ");
                        Console.WriteLine();
                        
                        // Full FFN
                        var ffnOut = block.Ffn.Forward(n2);
                        n2.Dispose();
                        
                        Console.Write($"  L00 FFN out[0][0..10]: ");
                        for (int k = 0; k < 10; k++) Console.Write($"{ffnOut.Data[k]:G6} ");
                        Console.WriteLine();
                        
                        var output = SharpMind.Core.Ops.TensorOps.Add(h1, ffnOut);
                        h1.Dispose();
                        ffnOut.Dispose();
                        hiddenState = output;
                        
                        Console.Write($"  L00 Output[0][0..20]: ");
                        for (int k = 0; k < 20; k++) Console.Write($"{output.Data[k]:G6} ");
                        Console.WriteLine();
                    }
                    else
                    {
                        using var normed1 = block.Norm1.Forward(hiddenState);
                        var attnOut = block.Attention.Forward(normed1, model.Ops, positionOffset: 0, causal: true, cache: caches[layerIdx]);
                        normed1.Dispose();
                        var h1 = SharpMind.Core.Ops.TensorOps.Add(hiddenState, attnOut);
                        attnOut.Dispose();
                        using var normed2 = block.Norm2.Forward(h1);
                        var ffnOut = block.Ffn.Forward(normed2);
                        normed2.Dispose();
                        var output = SharpMind.Core.Ops.TensorOps.Add(h1, ffnOut);
                        h1.Dispose();
                        ffnOut.Dispose();

                        if (layerIdx > 0) hiddenState.Dispose();
                        hiddenState = output;
                    }

                    // Check logit for [71486] from last token
                    double logit71486 = 0;
                    var lastToken = hiddenState.Data.Slice((seqLen - 1) * hiddenDim, hiddenDim);
                    for (int k = 0; k < hiddenDim; k++)
                        logit71486 += lastToken[k] * projW.Data[71486 * hiddenDim + k];

                    Console.WriteLine($"  L{layerIdx:D2}: hMax={MaxAbs(hiddenState.Data):G5} [71486]={logit71486,8:G5}");
                }

                // Final norm
                var finalNormed = model.FinalNorm.Forward(hiddenState);
                var fnLast = finalNormed.Data.Slice((seqLen - 1) * hiddenDim, hiddenDim);
                double finalLogit71486 = 0;
                for (int k = 0; k < hiddenDim; k++)
                    finalLogit71486 += fnLast[k] * projW.Data[71486 * hiddenDim + k];
                Console.WriteLine($"  FinalNorm: [71486]={finalLogit71486,8:G5}");
                Console.WriteLine($"  FinalNorm max={MaxAbs(finalNormed.Data):G5} meanAbs={MeanAbs(finalNormed.Data):G5}");

                // ── LM head projection ──
                using var normedFlat = finalNormed.Reshape(seqLen, hiddenDim);
                using var logits = model.Ops.MatMulWithBT(normedFlat, projW);
                Console.WriteLine($"Logits shape: [{logits.Shape[0]},{logits.Shape[1]}]");
                finalNormed.Dispose();
                hiddenState.Dispose();

                // Extract last token logits
                int vocabSize2 = Math.Min(logits.Shape[1], 151936);
                var lastLogits = new float[vocabSize2];
                logits.Data.Slice((seqLen - 1) * vocabSize2, vocabSize2).CopyTo(lastLogits);
                Console.WriteLine("SharpMind per-token logits:");
                for (int t = 0; t < seqLen; t++)
                {
                    Console.Write($"Token {t} (id={promptTokens[t]}):");
                    foreach (int cid in checkToks)
                    {
                        double sum = logits.Data[t * vocabSize2 + cid];
                        Console.Write($" [{cid}]={sum,8:G4}");
                    }
                    Console.WriteLine();
                }
                foreach (var c in caches) c.Dispose();
                logits.Dispose();
                model.Dispose();
                return lastLogits;
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
            return [.. indexed.Take(k)];
        }
    }
}
