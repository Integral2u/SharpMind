using System.Diagnostics;
using SharpMind.Core;
using SharpMind.Core.AgentTools;
using SharpMind.CUI.App;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Model;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Characterizes the real model's per-token forward cost on this machine so a
/// prefill fix targets the actual bottleneck: a large fixed per-call overhead
/// (workspace rent / layer plumbing) would make larger prefill chunks win big,
/// whereas per-token compute-bound scaling means only prompt-shrink + progress
/// help. Each probe creates a fresh generator (fresh KV cache + workspace) and
/// times a prefill of the given length plus a single decode step.
/// </summary>
public sealed class ModelSpeedProbeTests
{
    private const string ModelPath = @"C:\Users\tarra\SharpMind\Models\qwen2-0_5b-instruct-q8_0.gguf";
    private const string LogPath = @"C:\Users\tarra\AppData\Local\Temp\model_speed.log";

    private readonly ITestOutputHelper _output;

    public ModelSpeedProbeTests(ITestOutputHelper output) => _output = output;

    private static void Log(string message) =>
        File.AppendAllText(LogPath, $"{DateTime.Now:HH:mm:ss.fff} {message}{Environment.NewLine}");

    [Fact]
    public async Task Probe_ForwardCostScaling()
    {
        if (!File.Exists(ModelPath))
        {
            Log("SKIP: model not present");
            return;
        }

        Log("=== BEGIN speed probe (Release, Auto, parallel) ===");
        var options = new SessionOptions
        {
            ModelPath = ModelPath,
            HardwareTier = HardwareTier.Auto,
            UseParallelKernels = true,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
        };

        var load = await SessionLauncher.LoadModelAsync(options);
        Assert.True(load.Success, load.Error ?? "load failed");

        try
        {
            var loaded = load.Loaded!;
            Log($"loaded, layers={loaded.Model.Config.NumLayers} heads={loaded.Model.Config.NumHeads} kvHeads={loaded.Model.Config.NumKvHeads} dim={loaded.Model.Config.HiddenDim}");

            bool addBos = ModelMetaData.ResolveAddBos(loaded.Meta, loaded.Tokenizer.UseSentencePieceMerge);
            bool addEos = ModelMetaData.ResolveAddEos(loaded.Meta);

            foreach (int promptLen in new[] { 1, 4, 16, 64, 112, 128, 256, 512 })
            {
                var prompt = BuildPrompt(loaded.Tokenizer, promptLen);
                var sw = Stopwatch.StartNew();

                using var generator = new StandardGenerator<KVCacherBuilder>(
                    loaded.Model, loaded.Tokenizer, addBos, addEos, caches: null, seed: 1);
                generator.PrefillProgress = p => { };

                int fragments = 0;
                await foreach (var _ in generator.GenerateFromTokensAsync(
                    prompt, generation: new GenerationConfig { MaxNewTokens = 1 }))
                {
                    fragments++;
                    break;
                }

                long ms = sw.ElapsedMilliseconds;
                string line = $"prompt={prompt.Length,4} pkChunksFinish_ms={ms,5} msPerPromptToken={(double)ms / Math.Max(1, prompt.Length),7:F1} fragments={fragments}";
                Log(line);
                _output.WriteLine(line);
            }

            // Cheap GC settle so the numbers above aren't distorted by an earlier probe's shutdown.
            GC.Collect();
            GC.WaitForPendingFinalizers();
        }
        finally
        {
            load.Loaded?.Model?.Dispose();
        }
        Log("=== END speed probe ===");
    }

    private static int[] BuildPrompt(Tokenizer tokenizer, int targetTokens)
    {
        const string sentence = "The quick brown fox jumps over the lazy dog. ";
        var text = new System.Text.StringBuilder(targetTokens * 4);
        while (text.Length < targetTokens * 8)
            text.Append(sentence);
        var ids = new List<int>();
        foreach (int id in tokenizer.Encode(text.ToString(), addBos: false, addEos: false))
        {
            ids.Add(id);
            if (ids.Count >= targetTokens) break;
        }
        return [.. ids];
    }
}