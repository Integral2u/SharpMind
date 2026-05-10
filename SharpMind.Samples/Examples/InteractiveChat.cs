using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind;
using SharpMind.Inference;
using SharpMind.Inference.Chat;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;

namespace SharpMind.Samples.Examples;

/// <summary>
/// Interactive CLI chat using the full <see cref="ChatSession"/> pipeline:
/// real BPE tokenizer ? KV-cached transformer ? streaming token output.
///
/// Replaces the former <c>SimpleChatSession</c> which used a fake ASCII-level
/// tokenizer, re-ran the full context on every decode step (O(n²)), and
/// silently discarded the agent prompt.
///
/// Commands during chat:
///   quit   – exit
///   clear  – wipe conversation history, re-add system messages
/// </summary>
public static class InteractiveChat
{
    private static readonly string GgufModelName = "qwen2-0_5b-instruct-fp16";
    private static void Log(string msg) => Console.WriteLine(msg);

    public static async Task RunAsync()
    {
        
        Log("=== SharpMind Interactive Chat ===");
        Console.WriteLine("Commands: 'quit', 'clear'");
        Console.WriteLine();

        var hardware = DetectBestHardware();
Log($"Hardware: {hardware}");

        var baseDir       = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        var ggufPath      = Path.Combine(baseDir, $"{GgufModelName}.gguf");
        var tokenizerPath = Path.Combine(baseDir, $"{GgufModelName}.json");

        // -- GGUF existence check ------------------------------------------
        if (!File.Exists(ggufPath))
        {
            Log($"GGUF not found: {ggufPath}");
            return;
        }

        // -- Load metadata -------------------------------------------------
        var fileInfo = new FileInfo(ggufPath);
        Log($"\nGGUF: {ggufPath}  ({fileInfo.Length / 1_000_000.0:F1} MB)");

        int vocabSize = 128256, hiddenDim = 1536, numLayers = 24;
        int numHeads = 12, numKvHeads = 12, ffnDim = 6144, maxSeqLen = 2048;
        GgufLoader.GgufMeta meta;

        try
        {
            meta = GgufLoader.LoadMeta(ggufPath);
            Log($"  Loaded metadata: {meta.TensorCount} tensors");

            // DEBUG: Print all GGUF KV pairs to discover actual key names
            Console.WriteLine("[DEBUG] === GGUF METADATA KV PAIRS ===");
            foreach (var kv in meta.KvPairs)
                Console.WriteLine($"  {kv.Key} = {kv.Value}");
            Console.WriteLine("[DEBUG] === END KV PAIRS ===");

            // DEBUG: Print all tensor names and shapes - first 30 tensors for layout analysis
            Console.WriteLine("[DEBUG] === GGUF TENSOR SHAPES (first 30) ===");
            for (int i = 0; i < Math.Min(30, meta.Tensors.Count); i++)
            {
                var t = meta.Tensors[i];
                string shape = t.Shape != null ? string.Join("x", t.Shape) : "null";
                Console.WriteLine($"  [{i}] {t.Name}  dtype={t.Dtype}  shape={shape}");
            }
            Console.WriteLine("[DEBUG] === END TENSOR SHAPES ===");

            var embdInfo = meta.Tensors.FirstOrDefault(t =>
                t.Name.Contains("token_embd") && t.Name.Contains("weight"));

            if (embdInfo.Shape is { Length: >= 2 })
            {
                long d0 = embdInfo.Shape[0], d1 = embdInfo.Shape[1];
                if (d0 > d1) { vocabSize = (int)d0; hiddenDim = (int)d1; }
                else         { vocabSize = (int)d1; hiddenDim = (int)d0; }
            }
            else
            {
                hiddenDim = (int)meta.GetLong("llama.embedding_length", 1536);
                vocabSize = (int)meta.GetLong("vocab_size", 32000);
            }

            numLayers  = (int)meta.GetLong("llama.block_count", 24);
            // FIX: Check architecture-specific keys with fallbacks to llama.* and qwen2.*
            string arch = meta.GetString("general.architecture", "llama");
            string pref = arch + ".";  // "llama." or "qwen2." etc.
            
            // DEBUG: Print what GetLong returns
            Console.WriteLine($"[DEBUG] GetLong(qwen2.embedding_length) = {meta.GetLong("qwen2.embedding_length", 0)}");
            Console.WriteLine($"[DEBUG] GetLong(qwen2.attention.head_count) = {meta.GetLong("qwen2.attention.head_count", 0)}");
            Console.WriteLine($"[DEBUG] GetLong(qwen2.attention.head_count_kv) = {meta.GetLong("qwen2.attention.head_count_kv", 0)}");
            Console.WriteLine($"[DEBUG] GetLong(qwen2.feed_forward_length) = {meta.GetLong("qwen2.feed_forward_length", 0)}");
            Console.WriteLine($"[DEBUG] arch from GetString = '{arch}'");
            Console.WriteLine($"[DEBUG] pref = '{pref}'");
            
            // Try architecture-specific key, then llama., then tensor shape inference
            hiddenDim = (int)meta.GetLong(pref + "embedding_length", 
                meta.GetLong("llama.embedding_length", 
                    meta.GetLong("embedding", hiddenDim)));
            ffnDim = (int)meta.GetLong(pref + "feed_forward_length",
                meta.GetLong("llama.feed_forward_length", ffnDim));
            maxSeqLen = (int)meta.GetLong(pref + "context_length",
                meta.GetLong("llama.context_length", maxSeqLen));
            
            // Heads - try architecture then fall back to calculated from hiddenDim
            numHeads = (int)meta.GetLong(pref + "attention.head_count",
                meta.GetLong("llama.attention.head_count", numHeads));
            numKvHeads = (int)meta.GetLong(pref + "attention.head_count_kv",
                meta.GetLong("llama.attention.head_count_kv", numKvHeads));
            
            // MaxSeqLen also from direct key
            maxSeqLen = Math.Max(maxSeqLen, (int)meta.GetLong("llama.context_length", 2048));

            Log($"  Config: vocab={vocabSize}, hidden={hiddenDim}, layers={numLayers}");
            Log($"  Heads: {numHeads} (kv={numKvHeads}), ffn={ffnDim}, ctx={maxSeqLen}");
        }
        catch (Exception ex)
        {
            Log($"  Error loading metadata: {ex.Message}");
            return;
        }

        // -- Head alignment fixups (only warn, don't auto-adjust - keep original values) ----
        int origHeads = numHeads;
        if (hiddenDim % numHeads != 0)
        {
            Log($"[Warning] HiddenDim ({hiddenDim}) not divisible by NumHeads ({numHeads}).");
            // Try to find a divisor for heuristic suggestion only
            int suggested = numHeads;
            for (int h = numHeads; h > 0; h--)
                if (hiddenDim % h == 0) { suggested = h; break; }
            Log($"[Warning] Suggested NumHeads: {suggested} (but using {numHeads})");
        }
        if (numHeads % numKvHeads != 0)
        {
            Log($"[Warning] NumHeads ({numHeads}) not divisible by NumKvHeads ({numKvHeads}).");
            int suggested = numKvHeads;
            for (int kv = numKvHeads; kv > 0; kv--)
                if (numHeads % kv == 0) { suggested = kv; break; }
            Log($"[Warning] Suggested NumKvHeads: {suggested} (but using {numKvHeads})");
        }

        // -- Log effective HeadDim -----------------------------------------------
        Log($"  Effective: HeadDim={hiddenDim / numHeads}, KvGroupSize={numHeads / numKvHeads}");

        // -- Build model ---------------------------------------------------
        var modelConfig = new ModelConfig
        {
            VocabSize  = vocabSize,  HiddenDim  = hiddenDim,
            NumLayers  = numLayers,  NumHeads   = numHeads,
            NumKvHeads = numKvHeads, FfnDim     = ffnDim,
            MaxSeqLen  = maxSeqLen,
        };

        // DEBUG: Print model config values and calculated derived values
        Console.WriteLine("[DEBUG] === MODEL CONFIG ===");
        Console.WriteLine($"  VocabSize = {vocabSize}");
        Console.WriteLine($"  HiddenDim = {hiddenDim}");
        Console.WriteLine($"  NumLayers = {numLayers}");
        Console.WriteLine($"  NumHeads = {numHeads}");
        Console.WriteLine($"  NumKvHeads = {numKvHeads}");
        Console.WriteLine($"  FfnDim = {ffnDim}");
        Console.WriteLine($"  MaxSeqLen = {maxSeqLen}");
        Console.WriteLine($"  HeadDim = {hiddenDim / numHeads} (HiddenDim/NumHeads)");
        Console.WriteLine($"  KvGroupSize = {numHeads / numKvHeads} (NumHeads/NumKvHeads)");
        Console.WriteLine("[DEBUG] === END MODEL CONFIG ===");

        var sharpConfig = new SharpMindConfig
        {
            Activation = ActivationKind.SiLU,
            Gate       = GateKind.SwiGLU,
            Ffn        = FfnKind.Gated,
            Attention  = AttentionKind.MQA,
            Norm       = NormKind.RMSNorm,
            Arch       = ArchKind.Decoder,
            Hardware   = hardware,
        };

        Log("\nBuilding model...");
        GC.Collect(); GC.WaitForPendingFinalizers();
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Log($"  {model.ParameterCount / 1_000_000.0:F1}M parameters");

        Log("Loading weights...");
        GC.Collect(); GC.WaitForPendingFinalizers();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
        sw.Stop();
        Log($"  Done in {sw.ElapsedMilliseconds}ms");

        // -- Load tokenizer ------------------------------------------------
        if (!File.Exists(tokenizerPath))
        {
            Log($"[Error] tokenizer.json not found at: {tokenizerPath}");
            Log("  Download it from the model's HuggingFace page and place it");
            Log("  in the same folder as the GGUF file.");
            return;
        }

        Log("Loading tokenizer...");
        var tokenizer = Tokenizer.FromLlama(tokenizerPath);
        Log($"  Vocab size: {tokenizer.VocabSize}");

        // -- Create inference ops and chat session -------------------------
        Log("Creating inference ops...");
        var inferOps = InferenceOpsFactory.Create(sharpConfig, InferenceConfig.Default);

        await using var session = new ChatSession(model, tokenizer, inferOps)
        {
            MaxTokens   = 512,
            Temperature = 0.7f,
            TopK        = 40,
            TopP        = 0.9f,
        };

        string systemPrompt  = File.Exists("System.md") ? File.ReadAllText("System.md").Trim() : "";
        string agentPrompt   = File.Exists("Agent.md")  ? File.ReadAllText("Agent.md").Trim()  : "";
        string combinedSystem = BuildSystemPrompt(systemPrompt, agentPrompt);

        if (!string.IsNullOrEmpty(combinedSystem))
            session.AddMessage(ChatRole.System, combinedSystem);

        Log("\nChat ready! Say hello.\n");

        // -- Chat loop -----------------------------------------------------
        while (true)
        {
            Console.Write("You: ");
            var input = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(input))
                continue;

switch (input.Trim().ToLowerInvariant())
            {
                case "quit":
                    goto done;

                case "clear":
                    session.ClearHistory();
                    if (!string.IsNullOrEmpty(combinedSystem))
                        session.AddMessage(ChatRole.System, combinedSystem);
                    Console.WriteLine("[Context cleared]\n");
                    continue;
            }

Console.Write("Bot: ");
            sw.Restart();
            int tokensGenerated = 0;

            try
            {
                await foreach (var entry in session.GetResponseStreamAsync(input))
                {
                    if (entry.TextDelta is { Length: > 0 } delta)
                    {
                        Console.Write(delta);
                        tokensGenerated++;
                    }
                }
            }
            catch (Exception ex)
            {
                var msg = $"\n[Error] {ex.GetType().Name}: {ex.Message}";
                Console.WriteLine(msg);
                File.AppendAllText("chat_error.log", msg + Environment.NewLine);
            }

            double elapsedSec = sw.Elapsed.TotalSeconds;
            Console.WriteLine();
            Console.WriteLine(
                $"  [{tokensGenerated} tokens, {elapsedSec:F1}s, " +
                $"{tokensGenerated / Math.Max(elapsedSec, 0.001):F1} tok/s]");
            Console.WriteLine();
        }

        done:
        Console.WriteLine("Chat ended.");
    }

    /// <summary>
    /// Combines system and agent prompts into a single system message.
    /// Either may be empty; if both are present they are separated by a blank line.
    /// </summary>
    private static string BuildSystemPrompt(string system, string agent)
    {
        if (string.IsNullOrEmpty(system) && string.IsNullOrEmpty(agent))
            return string.Empty;
        if (string.IsNullOrEmpty(agent))  return system;
        if (string.IsNullOrEmpty(system)) return agent;
        return system + "\n\n" + agent;
    }

    private static HardwareTier DetectBestHardware()
    {
        if (Avx2.IsSupported) return HardwareTier.AVX2;
        if (Fma.IsSupported)  return HardwareTier.FMA;
        return HardwareTier.Scalar;
    }
}
