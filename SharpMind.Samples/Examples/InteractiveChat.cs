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
/// tokenizer, re-ran the full context on every decode step (O(n�)), and
/// silently discarded the agent prompt.
///
/// Commands during chat:
///   quit   � exit
///   clear  � wipe conversation history, re-add system messages
/// </summary>
public static class InteractiveChat
{
    private const string GgufModelName = "qwen2-0_5b-instruct-q4_k_m";

    public static async Task RunAsync()
    {
        Console.WriteLine("=== SharpMind Interactive Chat ===");
        Console.WriteLine("Commands: 'quit', 'clear'");
        Console.WriteLine();

        var hardware = DetectBestHardware();
        Console.WriteLine($"Hardware: {hardware}");

        var baseDir       = @"C:\Integral2u\source\repos\SharpMind\ExternalAssets";
        var ggufPath      = Path.Combine(baseDir, $"{GgufModelName}.gguf");
        var tokenizerPath = Path.Combine(baseDir, $"{GgufModelName}.json");

        // -- GGUF existence check ------------------------------------------
        if (!File.Exists(ggufPath))
        {
            Console.WriteLine($"GGUF not found: {ggufPath}");
            return;
        }

        // -- Load metadata -------------------------------------------------
        var fileInfo = new FileInfo(ggufPath);
        Console.WriteLine($"\nGGUF: {ggufPath}  ({fileInfo.Length / 1_000_000.0:F1} MB)");

        int vocabSize = 128256, hiddenDim = 1536, numLayers = 24;
        int numHeads = 12, numKvHeads = 12, ffnDim = 6144, maxSeqLen = 2048;
        GgufLoader.GgufMeta meta;

        try
        {
            meta = GgufLoader.LoadMeta(ggufPath);
            Console.WriteLine($"  Loaded metadata: {meta.TensorCount} tensors");

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
            numHeads   = (int)meta.GetLong("llama.attention.head_count", 12);
            numKvHeads = (int)meta.GetLong("llama.attention.head_count_kv", 12);
            ffnDim     = (int)meta.GetLong("llama.feed_forward_length", 6144);
            maxSeqLen  = (int)meta.GetLong("llama.context_length", 2048);

            Console.WriteLine($"  Config: vocab={vocabSize}, hidden={hiddenDim}, layers={numLayers}");
            Console.WriteLine($"  Heads: {numHeads} (kv={numKvHeads}), ffn={ffnDim}, ctx={maxSeqLen}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  Error loading metadata: {ex.Message}");
            return;
        }

        // -- Head alignment fixups -----------------------------------------
        if (hiddenDim % numHeads != 0)
        {
            Console.WriteLine($"[Warning] HiddenDim ({hiddenDim}) not divisible by NumHeads ({numHeads}). Adjusting...");
            for (int h = numHeads; h > 0; h--)
                if (hiddenDim % h == 0) { numHeads = h; break; }
            Console.WriteLine($"[Warning] Adjusted NumHeads to: {numHeads}");
        }
        if (numHeads % numKvHeads != 0)
        {
            Console.WriteLine($"[Warning] NumHeads ({numHeads}) not divisible by NumKvHeads ({numKvHeads}). Adjusting...");
            for (int kv = numKvHeads; kv > 0; kv--)
                if (numHeads % kv == 0) { numKvHeads = kv; break; }
            Console.WriteLine($"[Warning] Adjusted NumKvHeads to: {numKvHeads}");
        }

        // -- Build model ---------------------------------------------------
        var modelConfig = new ModelConfig
        {
            VocabSize  = vocabSize,  HiddenDim  = hiddenDim,
            NumLayers  = numLayers,  NumHeads   = numHeads,
            NumKvHeads = numKvHeads, FfnDim     = ffnDim,
            MaxSeqLen  = maxSeqLen,
        };
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

Console.WriteLine("\nBuilding model...");
        GC.Collect(); GC.WaitForPendingFinalizers();
        var model = ModelFactory.Create(modelConfig, sharpConfig);
        Console.WriteLine($"  {model.ParameterCount / 1_000_000.0:F1}M parameters");

        Console.WriteLine("Loading weights...");
        GC.Collect(); GC.WaitForPendingFinalizers();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        GgufLoader.LoadWeightsToModel(ggufPath, meta, model);
        sw.Stop();
        Console.WriteLine($"  Done in {sw.ElapsedMilliseconds}ms");

        // -- Load tokenizer ------------------------------------------------
        if (!File.Exists(tokenizerPath))
        {
            Console.WriteLine($"[Error] tokenizer.json not found at: {tokenizerPath}");
            Console.WriteLine("  Download it from the model's HuggingFace page and place it");
            Console.WriteLine("  in the same folder as the GGUF file.");
            return;
        }

        Console.WriteLine("Loading tokenizer...");
        var tokenizer = Tokenizer.FromLlama(tokenizerPath);
        Console.WriteLine($"  Vocab size: {tokenizer.VocabSize}");

        // -- Create inference ops and chat session -------------------------
        Console.WriteLine("Creating inference ops...");
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

        Console.WriteLine("\nChat ready! Say hello.\n");

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

            await foreach (var entry in session.GetResponseStreamAsync(input))
            {
                if (entry.TextDelta is { Length: > 0 } delta)
                {
                    Console.Write(delta);
                    tokensGenerated++;
                }
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
