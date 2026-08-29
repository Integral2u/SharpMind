using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using SharpMind.Core;
using SharpMind.Core.Training;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;
using SharpMind.GPU;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.LoRA;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;

namespace GpuTrainBench;

/// <summary>
/// Random token ids, already whitespace-joined. The point of the tool is the training step, so
/// the loader must never be what is measured: no file IO, no tokenizer, no shuffling.
/// </summary>
internal sealed class FixedTokenSource(IReadOnlyList<string> docs) : IDataSource
{
    public long? EstimatedCount => docs.Count;
    public string Description => "fixed in-memory token source";

    public async IAsyncEnumerable<string> ReadAsync([EnumeratorCancellation] CancellationToken ct = default)
    {
        foreach (var d in docs) { ct.ThrowIfCancellationRequested(); yield return d; await Task.Yield(); }
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

/// <summary>
/// Wall-clock cost of one GPU LoRA training step, and — with <c>--prof</c> — where inside the
/// step it goes. Exists so a kernel change can be judged against a number rather than a feeling.
///
/// Throughput and breakdown are measured in SEPARATE passes over the same warmed engine, because
/// <see cref="GpuStepProfiler"/> synchronises the device at every phase boundary and a step under
/// the profiler is meaningfully slower than a real one. Reporting both off a single pass is how
/// a profiled step ends up being quoted as throughput.
/// </summary>
internal static class Program
{
    /// <summary>SmolLM2-135M: the shape the GPU work has been tracked against since 2026-08.</summary>
    private static readonly ModelConfig Default = new()
    {
        VocabSize = 49152, HiddenDim = 576, NumLayers = 30, NumHeads = 9, NumKvHeads = 3,
        FfnDim = 1536, MaxSeqLen = 512, Architecture = "llama",
    };

    private const string Help = """
        GpuTrainBench — wall-clock cost of a GPU LoRA training step.

          --batch 1,2,4   batch sizes to sweep                   (default 1,2,4)
          --seq 256       sequence length                        (default 256)
          --warmup 2      untimed steps before measuring         (default 2)
          --steps 5       timed steps per batch size             (default 5)
          --layers 30     transformer layers                     (default 30, SmolLM2-135M)
          --flash         use the flash attention path instead of the materialised one
          --prof          also print a per-phase breakdown, measured in a separate pass

        Throughput is always measured with the profiler off; --prof adds a second pass for the
        breakdown, because synchronising at every phase slows the step it is reporting on.
        SM_PROF=1 in the environment implies --prof.
        """;

    private static long ParamCount(ModelConfig m) => (long)m.VocabSize * m.HiddenDim
        + (long)m.NumLayers * (2L * m.HiddenDim * m.HiddenDim
                             + 2L * m.HiddenDim * (m.NumKvHeads * m.HeadDim)
                             + 3L * m.HiddenDim * m.FfnDim);

    private static DataLoader MakeLoader(ModelConfig mc, int batch, int seq, int steps)
    {
        var rng = new Random(42);
        // PackingBatcher needs `batch` documents per step and the loader must not run dry
        // mid-measurement — a short run silently reports fewer steps than were asked for.
        int docs = Math.Max(96, (steps + 2) * batch);
        var text = Enumerable.Range(0, docs)
            .Select(_ => string.Join(' ', Enumerable.Range(0, seq - 1).Select(_ => rng.Next(3, mc.VocabSize))))
            .ToList();
        return new DataLoader(CleaningPipeline.From(new FixedTokenSource(text)),
            t => t.Split(' ').Select(int.Parse).ToArray(),
            new PackingBatcher(batchSize: batch, maxSeqLen: seq), maxBatches: 10_000);
    }

    private static async Task Run(ModelConfig mc, int batch, int seq, int warmup, int timed, bool flash, bool profile)
    {
        var sc = SharpMindConfig.Llama with { Hardware = HardwareTier.Auto };
        var weights = ModelFactory.CreateForTraining(mc, sc);
        WeightInitializer.InitializeRandomly(weights, 9001);
        using var model = ModelFactory.CreateTrainingTransformer(weights, sc);
        using var lora = new LoRAModel(model,
            new LoRAConfig { Rank = 8, TargetModules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"] },
            seed: 1);
        var ps = lora.LoRAParameters().ToList();

        var ops = TrainingOpsFactory.Create(sc);
        using var opt = new AdamW(ps, ops, lr: 1e-4f, weightDecay: 0f);
        using var engine = new GpuBackpropEngine(GpuDevice.Shared, model, ps, sc, batch, seq, flashAttention: flash);

        // Three phases in one run: warm up, time without the profiler, then — if asked — profile
        // without timing. The profiler starts off whatever SM_PROF said, so the timed pass is
        // never the profiled one.
        engine.Profiler.Enabled = false;
        int total = warmup + timed + (profile ? timed : 0);

        var cfg = new TrainConfig
        {
            TotalSteps = total, GradAccumSteps = 1, GradClipNorm = 1f,
            // LogInterval gates the onStep callback, not console output: at anything but 1 the
            // timing below sees a subset of the steps, and at 0 TrainLoop divides by it.
            LogInterval = 1, CheckpointInterval = 0, KeepRecent = -1,
            CheckpointDir = Path.Combine(Path.GetTempPath(), "sm-gputrainbench-ckpt"),
        };
        var loop = new TrainLoop(model, ps, MakeLoader(mc, batch, seq, total), opt,
            new ConstantScheduler(1e-4f), ops, smmConfig: sc, config: cfg, engine: engine);

        var times = new List<double>();
        var sw = Stopwatch.StartNew();
        long last = 0;
        int n = 0;
        await loop.RunAsync(onStep: _ =>
        {
            long now = sw.ElapsedTicks;
            double ms = (now - last) * 1000.0 / Stopwatch.Frequency;
            last = now;
            n++;
            if (n > warmup && n <= warmup + timed) times.Add(ms);
            // Reset as well as enable: the profiler is per-engine, but the warm-up and timed
            // steps of THIS engine would still be sitting in it.
            if (n == warmup + timed && profile) { engine.Profiler.Reset(); engine.Profiler.Enabled = true; }
        });

        if (times.Count == 0)
        {
            Console.WriteLine($"  b={batch}: only {n} of {total} steps ran — the loader went dry.");
            return;
        }
        times.Sort();
        double median = times[times.Count / 2];
        Console.WriteLine($"  {$"b={batch}",-8} {median,9:F1} ms/step {batch * seq / (median / 1000.0),9:F0} tok/s"
                        + $"   (min {times[0]:F1}, max {times[^1]:F1}, n={times.Count})");
        if (profile) Console.Write(engine.Profiler.Format($"b={batch} breakdown", "  "));
    }

    private static async Task<int> Main(string[] args)
    {
        if (args.Contains("--help") || args.Contains("-h")) { Console.WriteLine(Help); return 0; }

        string? Arg(string name)
        {
            int i = Array.IndexOf(args, name);
            return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
        }
        int Int(string name, int fallback) =>
            Arg(name) is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;

        int seq = Int("--seq", 256), warmup = Int("--warmup", 2), timed = Int("--steps", 5);
        bool flash = args.Contains("--flash");
        // SM_PROF is what GpuStepProfiler itself reads, so honour it here too rather than making
        // callers remember which of two switches this particular tool wants.
        bool profile = args.Contains("--prof") || GpuStepProfiler.EnabledByDefault;
        int[] batches = (Arg("--batch") ?? "1,2,4")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(int.Parse).ToArray();
        var mc = Default with
        {
            NumLayers = Int("--layers", Default.NumLayers),
            MaxSeqLen = Math.Max(seq, Default.MaxSeqLen),
        };

        Console.WriteLine(GpuDevice.Shared.Description);
        Console.WriteLine($"{ParamCount(mc) / 1e6:F0}M params, seq {seq}, LoRA r=8, "
                        + $"{(flash ? "flash" : "materialised")} attention, "
                        + $"{warmup} warm-up + {timed} timed{(profile ? $" + {timed} profiled" : "")} steps");
        foreach (int b in batches)
        {
            Console.WriteLine();
            await Run(mc, b, seq, warmup, timed, flash, profile);
        }
        return 0;
    }
}
