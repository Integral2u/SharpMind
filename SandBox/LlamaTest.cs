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

            // 5. Run prefill in LLamaSharp and extract all-token logits
            Console.WriteLine("\nRunning LLamaSharp prefill (all logits)...");
            var batch = new LLamaBatch();
            for (int i = 0; i < llmTokens.Length; i++)
                batch.Add(llmTokens[i], i, LLamaSeqId.Zero, logits: true);
            context.NativeHandle.Decode(batch);

            // Compare each token's logits
            Console.WriteLine("\n═══ Token-by-token logit comparison ═══");
            int[] checkToks = [71486, 32313, 9707, 13048, 40, 0, 1, 151646];
            for (int t = 0; t < llmTokens.Length; t++)
            {
                Span<float> llmLogits = context.NativeHandle.GetLogitsIth(t);
                Console.Write($"Token {t} (id={sharpTokens[t]}):");
                foreach (int cid in checkToks)
                    Console.Write($" [{cid}]={llmLogits[cid],8:G4}");
                Console.WriteLine();
            }

            // Also get last-token top-10
            Span<float> lastLlmLogits = context.NativeHandle.GetLogitsIth(llmTokens.Length - 1);
            var llmTop10 = GetTopK(lastLlmLogits, 10);
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

            var sharpConfig = SharpMind.SharpMindConfig.Qwen with { Hardware = HardwareTier.AVX2 };
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

                var sharpConfig = SharpMind.SharpMindConfig.Qwen with { Hardware = HardwareTier.Scalar };
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
                    double mean = 0; for (int i = 0; i < Math.Min(10, row.Length); i++) mean += row[i]; mean /= Math.Min(10, row.Length);
                    Console.WriteLine($"  token {checkToken} weight: min={min:G4} max={max:G4} mean={mean:G4}");
                }
                else
                {
                    var embedWeight = model.EmbeddingWeight;
                    var row = embedWeight.RowSpan(checkToken);
                    float min = float.MaxValue, max = float.MinValue;
                    for (int i = 0; i < row.Length; i++) { float v = row[i]; if (v < min) min = v; if (v > max) max = v; }
                    Console.WriteLine($"  (embedding) token {checkToken} weight: min={min:G4} max={max:G4}");
                }

                var caches = new KVCache[modelConfig.NumLayers];
                for (int i = 0; i < modelConfig.NumLayers; i++)
                    caches[i] = new KVCache(1, modelConfig.NumKvHeads, 128, modelConfig.HeadDim);

                using var input = SharpMind.Core.Tensors.Tensor<int>.From(promptTokens, 1, promptTokens.Length);

                // Forward returns [Batch=1, SeqLen, VocabSize]
                using var logits = model.Forward(input, caches);
                int seqLen = promptTokens.Length;
                int vocabSize = Math.Min(logits.Data.Length / seqLen, 151936);

                // Print per-token logits for comparison
                int[] checkToks = [71486, 32313, 9707, 13048, 40, 0, 1, 151646];
                Console.WriteLine("\nSharpMind per-token logits:");
                for (int t = 0; t < seqLen; t++)
                {
                    int offset = t * vocabSize;
                    Console.Write($"Token {t} (id={promptTokens[t]}):");
                    foreach (int cid in checkToks)
                        Console.Write($" [{cid}]={logits.Data[offset + cid],8:G4}");
                    Console.WriteLine();
                }

                // Return last-token logits for top-K comparison
                var lastLogits = new float[vocabSize];
                logits.Data.Slice((seqLen - 1) * vocabSize, vocabSize).CopyTo(lastLogits);
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
            return indexed.Take(k).ToList();
        }
    }
}
