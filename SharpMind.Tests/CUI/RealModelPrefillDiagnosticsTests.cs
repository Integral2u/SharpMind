using SharpMind.Core;
using SharpMind.Core.AgentTools;
using SharpMind.CUI.App;
using SharpMind.Inference;
using SharpMind.Inference.Agent;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Model.Format;
using SharpMind.Tokenization;
using Xunit;

namespace SharpMind.Tests.CUI;

/// <summary>
/// Headless diagnostics for the chunked-prefill stall observed on the real
/// model (HardwareTier.Auto + parallel kernels). Loads the GGUF file exactly
/// like the CUI does, then drives StandardGenerator directly with a long
/// synthetic prompt and logs per-chunk progress so a stall is attributable to
/// a precise phase (load, encode, chunk N of M, decode token K).
///
/// Each config variant is a separate test so a stuck ForwardLastLogits (which
/// cannot be interrupted — the loop is synchronous and token-checkpoint-free)
/// only stalls that one invocation.
/// </summary>
public sealed class RealModelPrefillDiagnosticsTests
{
    private const string ModelPath = @"C:\Users\tarra\SharpMind\Models\qwen2-0_5b-instruct-q8_0.gguf";
    private const string LogPath = @"C:\Users\tarra\AppData\Local\Temp\real_models.log";

    public static TheoryData<string, HardwareTier, bool> Configs => new()
    {
        { "AutoParallelOn", HardwareTier.Auto, true },   // CUI defaults — hung on wip
        { "AutoParallelOff", HardwareTier.Auto, false }, // isolate parallel kernels
        { "ScalarParallelOff", HardwareTier.Scalar, false }, // isolate HardwareTier.Auto
    };

    private static void Log(string message) =>
        File.AppendAllText(LogPath, $"{DateTime.Now:HH:mm:ss.fff} {message}{Environment.NewLine}");

    /// <summary>~1000-token synthetic prompt (16 chunks at MaxChunkLength=64).</summary>
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

    [Theory]
    [MemberData(nameof(Configs))]
    public async Task ChunkedPrefill_RealModel_Streams(string label, HardwareTier tier, bool parallel)
    {
        if (!File.Exists(ModelPath))
        {
            Log($"SKIP {label}: model not present");
            return; // dev-machine diagnostic; no GGUF shipped in-repo
        }

        var sw = System.Diagnostics.Stopwatch.StartNew();
        Log($"=== BEGIN {label} (tier={tier}, parallel={parallel}) ===");

        var options = new SessionOptions
        {
            ModelPath = ModelPath,
            HardwareTier = tier,
            UseParallelKernels = parallel,
            AgentsEnabled = false,
            FileAccess = ToolPermission.Always,
            NetworkAccess = ToolPermission.Always,
        };

        var load = await SessionLauncher.LoadModelAsync(options);
        Log($"load done success={load.Success} elapsed={sw.ElapsedMilliseconds}ms");
        Assert.True(load.Success, load.Error ?? "load failed");

        LoadedModel? loaded = null;
        try
        {
            loaded = load.Loaded!;
            Log($"maxseq={loaded.Model.Config.MaxSeqLen} kvHeads={loaded.Model.Config.NumKvHeads} heads={loaded.Model.Config.NumHeads} layers={loaded.Model.Config.NumLayers}");

            var promptIds = BuildPrompt(loaded.Tokenizer, targetTokens: 1200);
            Log($"prompt tokens={promptIds.Length}");

            bool addBos = ModelMetaData.ResolveAddBos(loaded.Meta, loaded.Tokenizer.UseSentencePieceMerge);
            bool addEos = ModelMetaData.ResolveAddEos(loaded.Meta);
            using var generator = new StandardGenerator<KVCacherBuilder>(
                loaded.Model, loaded.Tokenizer, addBos, addEos, caches: null, seed: 1);

            generator.PrefillProgress = p => Log($"prefill {p:P1} @ {sw.ElapsedMilliseconds}ms");

            var generation = new GenerationConfig { MaxNewTokens = 12 };
            var timeout = TimeSpan.FromSeconds(240);

            Log($"starting generate @ {sw.ElapsedMilliseconds}ms");
            var run = RunAsync(generator, promptIds, generation, sw);
            var completed = await Task.WhenAny(run, Task.Delay(timeout));
            Log($"whenany @ {sw.ElapsedMilliseconds}ms: completed={ReferenceEquals(completed, run)}");

            if (!ReferenceEquals(completed, run))
            {
                // The prefill loop can't be interrupted mid-chunk; leave it to
                // finish and simply record the stall. Decode path is
                // cancellation-aware, so cancel helps there.
                Log($"STALL suspected: no stream entry before {timeout.TotalSeconds:N0}s");
                Assert.Fail($"{label}: stalled — no streamed fragment within {timeout.TotalSeconds:N0}s.");
            }

            var (fragments, text) = run.Result;
            Log($"DONE fragments={fragments} text=\"{Escape(text)}\" total={sw.ElapsedMilliseconds}ms");
            Assert.True(fragments > 0, $"{label}: generation produced no streamed fragments.");
        }
        finally
        {
            // Give a stuck prefill a moment to breathe before the model dies.
            loaded?.Model?.Dispose();
        }

        Log($"=== END {label} ({sw.ElapsedMilliseconds}ms) ===");
    }

    private static async Task<(int Fragments, string Text)> RunAsync(
        StandardGenerator<KVCacherBuilder> generator,
        int[] promptIds,
        GenerationConfig generation,
        System.Diagnostics.Stopwatch sw)
    {
        var sb = new System.Text.StringBuilder();
        int fragments = 0;
        await foreach (var fragment in generator.GenerateFromTokensAsync(promptIds, generation: generation))
        {
            fragments++;
            sb.Append(fragment);
            Log($"fragment #{fragments} \"{Escape(fragment)}\" @ {sw.ElapsedMilliseconds}ms");
            if (fragments >= 6) break;
        }
        return (fragments, sb.ToString());
    }

    private static string Escape(string s) =>
        s.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
}