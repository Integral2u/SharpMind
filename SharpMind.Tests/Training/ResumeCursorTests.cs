using SharpMind.Core;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Data.Pipeline;
using SharpMind.Data.Pipeline.Stages;
using SharpMind.Data.Sources;
using SharpMind.Model;
using SharpMind.Model.Config;
using SharpMind.Training;
using SharpMind.Training.Optimizers;
using SharpMind.Training.Schedulers;

namespace SharpMind.Tests.Training;

/// <summary>
/// The resume data cursor: every checkpoint records how many batches the run had
/// consumed, and a resumed run skips that many so it trains on what comes NEXT
/// instead of silently re-training the head of the corpus it has already seen.
/// Skips reduce the remaining batch budget (the two runs together stay within
/// the configured total), survive a skip longer than one corpus pass, and the
/// data seed a cursor was computed against is enforced on resume.
/// </summary>
public sealed class ResumeCursorTests : IDisposable
{
    private readonly TempDirectory _dir = new();
    public void Dispose() => _dir.Dispose();

    private static ModelConfig Cfg => new()
    {
        VocabSize = 16,
        HiddenDim = 8,
        NumLayers = 1,
        NumHeads = 2,
        NumKvHeads = 2,
        FfnDim = 16,
        MaxSeqLen = 16,
    };

    private static int[] Tokenise(string s) => TestTokens.Encode(s, 16);

    private string WriteCorpus(int docs, string name = "corpus.txt")
        => _dir.Write(name,
            string.Join('\n', Enumerable.Range(0, docs).Select(i => $"the quick fox jumps over the lazy dog number {i}")));

    private DataLoader Loader(string path, int? maxBatches, int batchSize = 2)
    {
        var pipeline = CleaningPipeline.From(new TextFileSource(path)).Pipe(new NormaliseWhitespace());
        return new DataLoader(pipeline, Tokenise,
            new PackingBatcher(batchSize: batchSize, maxSeqLen: 16),
            maxBatches: maxBatches);
    }

    private static string Fingerprint(TrainingBatch batch)
        => string.Join(",", batch.TokenIds.Data.ToArray());

    private static async Task<List<string>> CollectAsync(DataLoader loader, int? maxBatches, int skip = 0)
    {
        var seen = new List<string>();
        await foreach (var batch in loader.LoadAsync(maxBatches: maxBatches, skipBatches: skip))
        {
            seen.Add(Fingerprint(batch));
            batch.Dispose();
        }
        return seen;
    }

    // ---- DataLoader: the skip itself ----

    [Fact]
    public async Task SkipBatches_EmitsTheTailOfTheUninterruptedStream()
    {
        string path = WriteCorpus(docs: 40);

        var full = await CollectAsync(Loader(path, maxBatches: 10), maxBatches: 10);
        var tail = await CollectAsync(Loader(path, maxBatches: 10), maxBatches: 10, skip: 4);

        Assert.Equal(10, full.Count);
        // A skip shrinks the budget, so the resumed run stays inside the
        // configured total rather than overshooting by the number skipped.
        Assert.Equal(6, tail.Count);
        Assert.Equal(full.Skip(4), tail);
    }

    [Fact]
    public async Task SkipLongerThanOnePass_StillEmitsTheRemainingBudget()
    {
        // Four docs and batchSize 2 fill exactly two rows per pass, so one corpus
        // pass yields 2 batches - a skip of 5 can never be satisfied by pass 1
        // alone. The pass that only absorbed the skip is not "empty"; without
        // counting it the stream would end having emitted nothing at all.
        string path = WriteCorpus(docs: 4, name: "small.txt");

        var batches = await CollectAsync(Loader(path, maxBatches: 10), maxBatches: 10, skip: 5);

        Assert.Equal(5, batches.Count);
    }

    [Fact]
    public async Task SkipCoveringTheWholeBudget_EmitsNothing()
    {
        string path = WriteCorpus(docs: 40);

        var batches = await CollectAsync(Loader(path, maxBatches: 4), maxBatches: 4, skip: 4);

        Assert.Empty(batches);
    }

    [Fact]
    public async Task SkipBeyondTheBudget_EmitsNothing()
    {
        // The skip already covers the entire budget, so there is nothing left to
        // emit - and nothing worth streaming the corpus just to discard.
        string path = WriteCorpus(docs: 200);

        var batches = await CollectAsync(Loader(path, maxBatches: 2), maxBatches: 2, skip: 10_000);

        Assert.Empty(batches);
    }

    // ---- TrainLoop: recording and applying the cursor ----

    [Fact]
    public async Task Checkpoint_RecordsBatchesConsumed()
    {
        string ckDir = Path.Combine(_dir.Path, "ck");
        Directory.CreateDirectory(ckDir);

        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 3,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = ckDir,
                KeepRecent = 5,
            },
            loaderBatches: 6);

        // GradAccumSteps = 1, so one step consumes exactly one batch.
        Assert.Equal("1", Checkpoint.ReadMeta(Path.Combine(ckDir, "step-0000001")).Extra["batchesConsumed"]);
        Assert.Equal("2", Checkpoint.ReadMeta(Path.Combine(ckDir, "step-0000002")).Extra["batchesConsumed"]);
        Assert.Equal("3", Checkpoint.ReadMeta(Path.Combine(ckDir, "step-0000003")).Extra["batchesConsumed"]);
    }

    [Fact]
    public async Task Resume_SkipsConsumedBatches_AndStaysWithinTheTotalBudget()
    {
        string corpus = WriteCorpus(docs: 40);
        string phase1Dir = Path.Combine(_dir.Path, "phase1");
        string phase2Dir = Path.Combine(_dir.Path, "phase2");

        // Phase 1: train two steps, checkpoints at every step.
        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 2,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = phase1Dir,
                KeepRecent = 5,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        string step2 = Path.Combine(phase1Dir, "step-0000002");
        Assert.Equal("2", Checkpoint.ReadMeta(step2).Extra["batchesConsumed"]);

        // Phase 2: resume at step 2 for two more steps. The cursor skips the two
        // batches phase 1 already trained on, so batchesSeen carries on at 4.
        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 4,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = phase2Dir,
                KeepRecent = 5,
                ResumeFrom = step2,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        var resumed = Checkpoint.ReadMeta(Path.Combine(phase2Dir, "step-0000004-final"));
        Assert.Equal(4, resumed.Step);
        Assert.Equal("4", resumed.Extra["batchesConsumed"]);
    }

    /// <summary>
    /// The cursor is only useful if it reproduces the same stream: an
    /// uninterrupted 4-step run and a 2+2 split must account for exactly the
    /// same tokens, which cannot happen if the resumed leg re-trained the head
    /// of the corpus (or replayed it from a different position).
    /// </summary>
    [Fact]
    public async Task Resume_ContinuesTokensSeenWhereAnUninterruptedRunWouldBe()
    {
        string corpus = WriteCorpus(docs: 40);
        string straightDir = Path.Combine(_dir.Path, "straight");
        string split1 = Path.Combine(_dir.Path, "split1");

        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 4,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = straightDir,
                KeepRecent = 5,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 2,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = split1,
                KeepRecent = 5,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 4,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = Path.Combine(_dir.Path, "split2"),
                KeepRecent = 5,
                ResumeFrom = Path.Combine(split1, "step-0000002"),
            },
            loaderBatches: 8,
            corpusPath: corpus);

        string straight = Checkpoint.ReadMeta(Path.Combine(straightDir, "step-0000004-final"))
            .Extra["tokensSeen"];
        string split = Checkpoint.ReadMeta(Path.Combine(_dir.Path, "split2", "step-0000004-final"))
            .Extra["tokensSeen"];

        Assert.Equal(straight, split);
    }

    [Fact]
    public async Task ResumeDataCursorDisabled_DoesNotSkip()
    {
        string corpus = WriteCorpus(docs: 40);
        string phase1Dir = Path.Combine(_dir.Path, "p1");

        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 2,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = phase1Dir,
                KeepRecent = 5,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        string step2 = Path.Combine(phase1Dir, "step-0000002");

        // An incremental run feeds only new files, so its stream is NOT the one
        // the cursor was recorded against - the cursor must be ignored and the
        // batcher start from the top of the delta.
        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 3,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = Path.Combine(_dir.Path, "p2"),
                KeepRecent = 5,
                ResumeFrom = step2,
                ResumeDataCursor = false,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        var resumed = Checkpoint.ReadMeta(Path.Combine(_dir.Path, "p2", "step-0000003-final"));
        Assert.Equal(3, resumed.Step);
        // Only the batch this leg trained on: nothing was skipped.
        Assert.Equal("1", resumed.Extra["batchesConsumed"]);
    }

    [Fact]
    public async Task Resume_DataSeedMismatch_Throws()
    {
        string corpus = WriteCorpus(docs: 40);
        string phase1Dir = Path.Combine(_dir.Path, "p1");

        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 2,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = phase1Dir,
                KeepRecent = 5,
                DataSeed = 1234,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        string step2 = Path.Combine(phase1Dir, "step-0000002");
        Assert.Equal("1234", Checkpoint.ReadMeta(step2).Extra["dataSeed"]);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 4,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = Path.Combine(_dir.Path, "p2"),
                KeepRecent = 5,
                ResumeFrom = step2,
                DataSeed = 42,
            },
            loaderBatches: 8,
            corpusPath: corpus));

        Assert.Contains("seed", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task Resume_DataSeedMatch_IsAccepted()
    {
        string corpus = WriteCorpus(docs: 40);
        string phase1Dir = Path.Combine(_dir.Path, "p1");

        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 2,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = phase1Dir,
                KeepRecent = 5,
                DataSeed = 1234,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 3,
                GradAccumSteps = 1,
                CheckpointInterval = 1,
                CheckpointDir = Path.Combine(_dir.Path, "p2"),
                KeepRecent = 5,
                ResumeFrom = Path.Combine(phase1Dir, "step-0000002"),
                DataSeed = 1234,
            },
            loaderBatches: 8,
            corpusPath: corpus);

        var resumed = Checkpoint.ReadMeta(Path.Combine(_dir.Path, "p2", "step-0000003-final"));
        Assert.Equal(3, resumed.Step);
        Assert.Equal("3", resumed.Extra["batchesConsumed"]);
    }

    // ---- A6: pause semantics ----

    /// <summary>
    /// A6's done-when: cancelling a run writes an interrupted checkpoint at the
    /// step it reached, and resuming from it logs step+1 next - the run picks up
    /// where it stopped rather than replaying the batch it had already trained on
    /// (the cursor in Extra is what makes that true).
    /// </summary>
    [Fact]
    public async Task Cancel_ThenResume_NextLoggedStepIsStepPlusOne()
    {
        string corpus = WriteCorpus(docs: 40);
        string phase1Dir = Path.Combine(_dir.Path, "p1");

        var cts = new CancellationTokenSource();
        var steps = new List<int>();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 10,
                GradAccumSteps = 1,
                LogInterval = 1,
                CheckpointInterval = 1,
                CheckpointDir = phase1Dir,
                KeepRecent = 5,
            },
            loaderBatches: 20,
            corpusPath: corpus,
            cancellationToken: cts.Token,
            onStep: r =>
            {
                steps.Add(r.Step);
                if (r.Step == 3) cts.Cancel();
            }));

        Assert.Equal(new[] { 1, 2, 3 }, steps);

        // The cancel overwrote step-0000003 with an interrupted checkpoint that
        // carries the cursor for the three batches already trained on.
        var interrupted = Checkpoint.ReadMeta(Path.Combine(phase1Dir, "step-0000003"));
        Assert.Equal(3, interrupted.Step);
        Assert.Equal("interrupted", interrupted.Note);
        Assert.Equal("3", interrupted.Extra["batchesConsumed"]);

        // Resume: the next logged step is 4 (3 + 1), and no batch is replayed.
        var resumedSteps = new List<int>();
        await RunLoopAsync(
            new TrainConfig
            {
                TotalSteps = 4,
                GradAccumSteps = 1,
                LogInterval = 1,
                CheckpointInterval = 1,
                CheckpointDir = Path.Combine(_dir.Path, "p2"),
                KeepRecent = 5,
                ResumeFrom = Path.Combine(phase1Dir, "step-0000003"),
            },
            loaderBatches: 20,
            corpusPath: corpus,
            onStep: r => resumedSteps.Add(r.Step));

        Assert.Equal(new[] { 4 }, resumedSteps);
        var final = Checkpoint.ReadMeta(Path.Combine(_dir.Path, "p2", "step-0000004-final"));
        Assert.Equal("4", final.Extra["batchesConsumed"]);
    }

    // ---- harness ----

    private async Task RunLoopAsync(
        TrainConfig config,
        int loaderBatches,
        string? corpusPath = null,
        CancellationToken cancellationToken = default,
        Action<TrainStepResult>? onStep = null)
    {
        var sharpConfig = SharpMindConfig.Gpt with { Hardware = HardwareTier.Scalar };
        var weights = ModelFactory.CreateForTraining(Cfg, sharpConfig);
        if (config.ResumeFrom is null)
            WeightInitializer.InitializeRandomly(weights, 1234);

        using var model = ModelFactory.CreateTrainingTransformer(weights, sharpConfig);
        var loader = Loader(corpusPath ?? WriteCorpus(docs: 40), loaderBatches);

        var loop = new TrainLoop(
            model,
            model.Parameters(),
            loader,
            new AdamW(model.Parameters(), lr: 0.01f, weightDecay: 0f),
            new ConstantScheduler(0.01f),
            TrainingOpsFactory.Create(sharpConfig),
            smmConfig: sharpConfig,
            config: config);

        await loop.RunAsync(onStep: onStep, cancellationToken: cancellationToken);
    }
}
