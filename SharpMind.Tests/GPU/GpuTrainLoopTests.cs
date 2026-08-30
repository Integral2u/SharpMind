using System.Runtime.CompilerServices;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Sources;
using SharpMind.GPU;
using SharpMind.Training;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;
using Xunit;

namespace SharpMind.Tests.GPU;

[Collection("GPU")]
public sealed class GpuTrainLoopTests
{
    /// <summary>In-memory <see cref="IDataSource"/> over a fixed list of pre-tokenised "documents"
    /// (space-separated token ids) — no disk I/O, fully deterministic.</summary>
    private sealed class FixedTokenSource(IReadOnlyList<string> docs) : IDataSource
    {
        public long? EstimatedCount => docs.Count;
        public string Description => "fixed in-memory token source";
        public async IAsyncEnumerable<string> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            foreach (var doc in docs)
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return doc;
                await Task.Yield();
            }
        }
        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }

    /// <summary>
    /// 4 docs of exactly S-1 tokens each (+ PackingBatcher's own EOS) fill batchSize=2 rows with
    /// zero padding, so one pass yields exactly 2 batches [A, B] and, with GradAccumSteps = 2,
    /// every training step sees that same [A, B] pair regardless of where in the stream it starts.
    /// That periodicity is what makes <see cref="GpuLoop_Resumed_MatchesCpuLoop"/> an
    /// apples-to-apples comparison: the resumed run's fresh loader restart replays the identical
    /// per-step data as the uninterrupted run, so only the checkpoint/device round-trip can make
    /// their final parameters differ. Token ids stay in [3, VocabSize) so none collide with
    /// PackingBatcher's own EOS(2)/pad(0). Batch shape is exactly (B, S) — the shape
    /// GpuBackpropEngine was constructed with; a mismatched shape throws by design.
    /// </summary>
    private static DataLoader MakeLoader()
    {
        var rng = new Random(42);
        int vocab = GpuBackpropEngineTests.Cfg.VocabSize;
        var docs = Enumerable.Range(0, 4)
            .Select(_ => string.Join(' ', Enumerable.Range(0, GpuBackpropEngineTests.S - 1).Select(__ => rng.Next(3, vocab))))
            .ToList();
        var pipeline = CleaningPipeline.From(new FixedTokenSource(docs));
        return new DataLoader(pipeline, s => s.Split(' ').Select(int.Parse).ToArray(),
            new PackingBatcher(batchSize: GpuBackpropEngineTests.B, maxSeqLen: GpuBackpropEngineTests.S),
            maxBatches: 20);
    }

    [Fact]
    public async Task GpuLoop_WithAccumulationClipAndCheckpoint_MatchesCpuLoop()
    {
        const float clipNorm = 1f;
        async Task<(List<float> losses, List<float> gradNorms, float[] finalB)> Run(bool gpu, string dir)
        {
            var (model, lora, ps, sc) = GpuBackpropEngineTests.Fixture();
            using var _ = model; using var __ = lora;
            var ops = TrainingOpsFactory.Create(sc);
            using var opt = new AdamW(ps, ops, lr: 1e-2f, weightDecay: 0f);
            ITrainingEngine? engine = gpu ? new GpuBackpropEngine(GpuTestDevice.Device, model, ps, sc, GpuBackpropEngineTests.B, GpuBackpropEngineTests.S) : null;
            var cfg = new TrainConfig { TotalSteps = 4, GradAccumSteps = 2, GradClipNorm = clipNorm, LogInterval = 1, CheckpointInterval = 2, CheckpointDir = dir, KeepRecent = 5 };
            var loop = new TrainLoop(model, ps, MakeLoader(), opt, new ConstantScheduler(1e-2f), ops, smmConfig: sc, config: cfg, engine: engine);
            var losses = new List<float>();
            var gradNorms = new List<float>();
            await loop.RunAsync(onStep: r => { losses.Add(r.Loss); gradNorms.Add(r.GradNorm); });
            engine?.Dispose();
            return (losses, gradNorms, ps.Last().Data.Data.ToArray());
        }
        string root = Path.Combine(Path.GetTempPath(), "sm-gpu-loop-" + Guid.NewGuid().ToString("N"));
        try
        {
            var cpu = await Run(false, Path.Combine(root, "cpu"));
            var gpu = await Run(true, Path.Combine(root, "gpu"));
            Assert.Equal(cpu.losses.Count, gpu.losses.Count);
            // Measured: exact-zero delta (< 5e-7) at this shape/seed. 1e-4 keeps ~200x headroom
            // above that floor — tight enough to catch a real regression, loose enough not to
            // flake on a different accelerator backend (this laptop has no NVIDIA GPU: OpenCL,
            // tiled GEMM, a different summation order than CUDA would use).
            for (int i = 0; i < cpu.losses.Count; i++) Assert.True(Math.Abs(cpu.losses[i] - gpu.losses[i]) < 1e-4, $"step {i}: {cpu.losses[i]} vs {gpu.losses[i]}");
            // Measured max rel err 6.13e-7, matching GpuBackpropEngineTests' established 1e-4
            // floor for this shape (the tolerance those tests already assert LoRA grads at).
            GpuTestDevice.AssertClose(cpu.finalB, gpu.finalB, 1e-4, "final lora_B");
            Assert.True(Directory.Exists(Path.Combine(root, "gpu", "step-0000002")), "checkpoint written");
            // Prove clipping actually engaged, not just configured: GradNorm is the PRE-clip
            // global norm (Gradients.ClipGlobalNorm's own doc comment), so this model/seed must
            // produce at least one step whose raw gradient norm exceeds clipNorm=1, or the clip
            // is inert and this fact isn't proving what its name claims. Measured per-step norms:
            // CPU [1.237, 1.496, 1.368, 0.945], GPU [1.237, 1.496, 1.368, 0.945] — steps 1-3 clip
            // (norm > 1), step 4 doesn't (0.945 < 1, gradient shrank as training converged).
            Assert.True(cpu.gradNorms.Max() > clipNorm, $"CPU pre-clip norm never exceeded {clipNorm}: {string.Join(",", cpu.gradNorms)}");
            Assert.True(gpu.gradNorms.Max() > clipNorm, $"GPU pre-clip norm never exceeded {clipNorm}: {string.Join(",", gpu.gradNorms)}");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    /// <summary>
    /// Proves <c>Checkpoint.Load</c>'s host-side <c>Parameter.Data</c> write survives to the
    /// device via <c>GpuLinear.SyncLoRAToDevice</c> at the next step.
    ///
    /// The oracle is a CPU run, not a second GPU run: an earlier fault-injection pass (commenting
    /// out the per-step <c>SyncLoRAToDevice</c> call in <c>GpuBackpropEngine.ForwardBackward</c>)
    /// showed that comparing a resumed GPU run against a fresh "uninterrupted" GPU run does NOT
    /// catch that bug. Both sides are built from <see cref="GpuBackpropEngineTests.Fixture"/>,
    /// which is fully deterministic (fixed seeds); with sync disabled, every engine's device
    /// buffers freeze at that SAME construction-time value forever, in every phase, whether or
    /// not <c>Checkpoint.Load</c> ran first. The gradient computed at every step then becomes a
    /// constant, so AdamW's trajectory becomes a pure function of (constant gradient, step count)
    /// — perfectly reproducible across a split resume even though the device never once reflects
    /// the checkpoint. CPU has no device buffer at all, so it cannot share that failure mode: its
    /// weights genuinely evolve host-side every step, so if the GPU device silently stops
    /// tracking the (correctly-evolving) host weights, GPU's gradients start being computed
    /// against the wrong weights and its trajectory measurably diverges from CPU's — exactly the
    /// divergence <see cref="GpuLoop_WithAccumulationClipAndCheckpoint_MatchesCpuLoop"/> already
    /// demonstrates starting at the second optimizer step once sync is disabled.
    ///
    /// <see cref="MakeLoader"/>'s [A, B] periodicity (documented there) is what makes this a fair
    /// comparison at all: CPU trains 4 steps continuously off ONE loader, while the resumed GPU
    /// run's phase 2 restarts a FRESH loader after loading the step-2 checkpoint — periodicity
    /// guarantees both see the identical [A, B] pair at every step regardless of the restart, so
    /// only the checkpoint/device round-trip — not a data-order mismatch — can separate them.
    /// </summary>
    [Fact]
    public async Task GpuLoop_Resumed_MatchesCpuLoop()
    {
        const float clipNorm = 1f;
        async Task<(float[] finalB, List<float> gradNorms)> Run(TrainConfig cfg, bool gpu)
        {
            var (model, lora, ps, sc) = GpuBackpropEngineTests.Fixture();
            using var _ = model; using var __ = lora;
            var ops = TrainingOpsFactory.Create(sc);
            using var opt = new AdamW(ps, ops, lr: 1e-2f, weightDecay: 0f);
            ITrainingEngine? engine = gpu ? new GpuBackpropEngine(GpuTestDevice.Device, model, ps, sc, GpuBackpropEngineTests.B, GpuBackpropEngineTests.S) : null;
            var loop = new TrainLoop(model, ps, MakeLoader(), opt, new ConstantScheduler(1e-2f), ops, smmConfig: sc, config: cfg, engine: engine);
            var gradNorms = new List<float>();
            await loop.RunAsync(onStep: r => gradNorms.Add(r.GradNorm));
            engine?.Dispose();
            return (ps.Last().Data.Data.ToArray(), gradNorms);
        }

        string root = Path.Combine(Path.GetTempPath(), "sm-gpu-resume-" + Guid.NewGuid().ToString("N"));
        try
        {
            var cpu = await Run(new TrainConfig { TotalSteps = 4, GradAccumSteps = 2, GradClipNorm = clipNorm, LogInterval = 1, CheckpointInterval = 0 }, gpu: false);

            string resumeDir = Path.Combine(root, "resume");
            var phase1 = await Run(new TrainConfig { TotalSteps = 2, GradAccumSteps = 2, GradClipNorm = clipNorm, LogInterval = 1, CheckpointInterval = 2, CheckpointDir = resumeDir, KeepRecent = 5 }, gpu: true);
            string step2Dir = Path.Combine(resumeDir, "step-0000002");
            Assert.True(Directory.Exists(step2Dir), "step-2 checkpoint written");

            var resumed = await Run(new TrainConfig { TotalSteps = 4, GradAccumSteps = 2, GradClipNorm = clipNorm, LogInterval = 1, CheckpointInterval = 2, CheckpointDir = resumeDir, KeepRecent = 5, ResumeFrom = step2Dir }, gpu: true);

            // Measured max rel err 6.13e-7 — identical to the uninterrupted comparison above,
            // as expected since both trace the same data/init. Same 1e-4 floor as that test.
            GpuTestDevice.AssertClose(cpu.finalB, resumed.finalB, 1e-4, "resumed final lora_B vs CPU oracle");
            // Same clipping proof as the loop test above, for both legs of the split resume run:
            // GradNorm is the PRE-clip norm, so each leg needs a step over clipNorm=1 or the clip
            // never engaged. Measured: phase1 (steps 1-2) [1.237, 1.496], resumed (steps 3-4,
            // post-checkpoint) [1.368, 0.945] — matches the uninterrupted run's steps 3-4 exactly,
            // as the [A, B] periodicity predicts (see MakeLoader/GpuLoop_Resumed_MatchesCpuLoop docs).
            Assert.True(phase1.gradNorms.Max() > clipNorm, $"phase-1 pre-clip norm never exceeded {clipNorm}: {string.Join(",", phase1.gradNorms)}");
            Assert.True(resumed.gradNorms.Max() > clipNorm, $"resumed pre-clip norm never exceeded {clipNorm}: {string.Join(",", resumed.gradNorms)}");
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
