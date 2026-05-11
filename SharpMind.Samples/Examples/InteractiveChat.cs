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

        // -- Single-pass loader: GGUF + optional tokenizer -------------------
        GgufMeta meta;
        ModelConfig modelConfig;
        Tokenizer? tokenizer;
        string? chatTemplate;

        try
        {
            GgufLoader.LoadDetails(ggufPath, tokenizerPath, out meta, out modelConfig, out tokenizer);
            chatTemplate = meta.GetChatTemplate();
            Log("  Single-pass loader succeeded");
        }
        catch (Exception ex)
        {
            Log($"Error loading: {ex.Message}");
            return;
        }
        
        int bosId = meta.GetSpecialTokenId("bos");
        int eosId = meta.GetSpecialTokenId("eos");

        Log($"  Effective: HeadDim={modelConfig.HeadDim}, KvGroupSize={modelConfig.NumHeads / modelConfig.NumKvHeads}");

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

        // -- Create inference ops -------------------------
        Log("Creating inference ops...");
        var inferOps = InferenceOpsFactory.Create(sharpConfig, InferenceConfig.Default);

        await using var session = new ChatSession(model, tokenizer, inferOps, bosId, eosId, chatTemplate)
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
