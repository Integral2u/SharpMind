using SharpMind.Core;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Model;
using SharpMind.Core.Training;
using SharpMind.Training.Loss;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Autograd;

namespace SharpMind.Training;

/// <summary>
/// The training loop. Connects data â†’ model â†’ loss â†’ backward â†’ optimizer â†’ checkpoint.
///
/// Usage:
/// <code>
/// var loop = new TrainLoop(
///     model:      model,
///     parameters: model.Parameters(),
///     loader:     dataLoader,
///     optimizer:  new AdamW(model.Parameters(), lr: 3e-4f),
///     scheduler:  new CosineWithWarmup(3e-4f, 3e-5f, warmup: 2000, decay: 10000),
///     config:     new TrainConfig { TotalSteps = 10_000 });
///
/// await loop.RunAsync(onStep: r => Console.WriteLine($"step={r.Step} loss={r.Loss:F4}"));
/// </code>
/// </summary>
public sealed class TrainLoop
{
    private readonly List<Parameter>       _parameters;
    private readonly DataLoader            _loader;
    private readonly IOptimizer            _optimizer;
    private readonly IScheduler            _scheduler;
    private readonly TrainConfig           _config;
    private readonly ILoss<int>            _loss;
    private readonly TrainingOps           _ops;
    private readonly GradientMapping       _mapping;
    private readonly ITrainingEngine       _engine;
    private readonly int                   _vocabSize;

    /// <param name="engine">
    /// The compute engine for forward + backward. Null selects the CPU reference
    /// engine; an accelerator plugin supplies its own. An engine you pass in remains
    /// yours to dispose â€” the loop never disposes it. The default CPU engine (used
    /// when this is null) lives with the loop instead.
    /// </param>
    public TrainLoop(
        Transformer           model,
        IEnumerable<Parameter> parameters,
        DataLoader             loader,
        IOptimizer             optimizer,
        IScheduler             scheduler,
        TrainingOps            ops,
        ILoss<int>?            loss      = null,
        GradientMapping?       mapping   = null,
        SharpMindConfig?       smmConfig = null,
        TrainConfig?           config    = null,
        ITrainingEngine?       engine    = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(scheduler);

        _parameters = [.. parameters];
        _loader     = loader;
        _optimizer  = optimizer;
        _scheduler  = scheduler;
        _config     = config ?? new TrainConfig();
        _loss       = loss ?? new CrossEntropyLoss(labelSmoothing: _config.LabelSmoothing);
        _mapping    = mapping ?? GradientMappingFactory.Create(smmConfig ?? new SharpMindConfig());
        _ops        = ops;
        _vocabSize  = model.Config.VocabSize;

        if (_config.QuantAwareTraining is { } qat)
            model.EnableQuantAwareTraining(qat);

        _engine     = engine ?? new CpuTrainingEngine(model, _mapping, _parameters, smmConfig ?? new SharpMindConfig(), _loss);
    }

    /// <summary>
    /// Runs the training loop for <see cref="TrainConfig.TotalSteps"/> steps.
    /// <paramref name="onStep"/> is called after each optimizer step with loss
    /// and diagnostic info â€” use for logging, early stopping, or eval triggers.
    /// <paramref name="progress"/> reports 0..1 after each optimizer step.
    /// <paramref name="onCheckpoint"/> is called with the full path of each
    /// saved checkpoint directory (rolling and final), after retention pruning.
    /// </summary>
    public async Task RunAsync(
        Action<TrainStepResult>? onStep            = null,
        IProgress<float>?        progress           = null,
        Action<string>?          onCheckpoint       = null,
        CancellationToken cancellationToken = default,
        Func<int, EvalResult>?   evaluate           = null)
    {
        int   startStep   = Math.Max(0, _config.StartStep);
        int   skipBatches = 0;
        long  priorTokens = 0;
        float bestValLoss = float.PositiveInfinity;

        if (_config.ResumeFrom is not null)
        {
            var meta = Checkpoint.Load(_config.ResumeFrom, _parameters, _optimizer);
            startStep = meta.Step;

            // Vocabulary guard: the weight-tied embedding is shape-bound to the
            // vocabulary it was built with, so a checkpoint from a different vocab
            // cannot be resumed as-is. Report that clearly here rather than letting
            // the binary loader throw a bare shape mismatch naming 'EmbeddingWeight'.
            if (_config.ExpectedVocabSize > 0
                && meta.Extra.TryGetValue("vocabSize", out var recordedVocab)
                && int.TryParse(recordedVocab, System.Globalization.NumberStyles.Integer,
                       System.Globalization.CultureInfo.InvariantCulture, out int recordedVocabSize)
                && recordedVocabSize != _config.ExpectedVocabSize)
            {
                throw new InvalidOperationException(
                    $"Checkpoint under '{_config.ResumeFrom}' was trained with vocabulary size " +
                    $"{recordedVocabSize}, but this run's model is built for {_config.ExpectedVocabSize}. " +
                    "The embedding table cannot be resumed across a vocabulary change â€” use a warm " +
                    $"restart ({nameof(TrainConfig.WarmRestartFrom)}) to reuse these weights and grow the " +
                    "tokenizer, or start fresh.");
            }

            // Data cursor: where the interrupted run stopped in the batch stream.
            // Restoring it means the resumed run trains on what comes NEXT instead
            // of silently re-training the head of the corpus it has already seen.
            // The cursor is only meaningful for the same data: if the checkpoint's
            // recorded fingerprint differs from this run's corpus, skip nothing
            // (replaying a stale count would walk far into unrelated batches).
            if (_config.ResumeDataCursor)
            {
                bool sameData = _config.DataFingerprint is null
                    || !meta.Extra.TryGetValue("dataFingerprint", out var recordedFingerprint)
                    || string.Equals(recordedFingerprint, _config.DataFingerprint, StringComparison.Ordinal);
                if (sameData)
                {
                    skipBatches = ReadExtraInt(meta.Extra, "batchesConsumed");
                    GuardDataSeed(meta.Extra);
                }
            }

            priorTokens = ReadExtraLong(meta.Extra, "tokensSeen");
            if (ReadExtraFloat(meta.Extra, "bestValLoss") is { } priorBest && float.IsFinite(priorBest))
                bestValLoss = priorBest;
        }

        int   step        = startStep;
        int   accumCount  = 0;
        float accumLoss   = 0f;
        int   stepTokens  = 0;
        long  tokensSeen  = priorTokens;
        int   batchesSeen = skipBatches;

        // Captures the live cursor/telemetry so every checkpoint - rolling,
        // best, interrupted or final - records where the run actually is.
        Dictionary<string, string> Extra(float? valLoss)
        {
            var extra = BuildExtra(tokensSeen, batchesSeen, bestValLoss, valLoss, _vocabSize,
                _config.DataFingerprint);
            WriteDataSeed(extra);
            return extra;
        }

        try
        {
            await foreach (var batch in _loader.LoadAsync(
                skipBatches: skipBatches, cancellationToken: cancellationToken))
            {
                cancellationToken.ThrowIfCancellationRequested();
                batchesSeen++;

                // Zero gradients at the start of each accumulation window
                if (accumCount == 0) _optimizer.ZeroGrad();

                var sw = System.Diagnostics.Stopwatch.StartNew();

                // Forward pass
                float batchLoss = ForwardBackward(batch, cancellationToken);
                accumLoss += batchLoss;
                int batchTokens = batch.RealTokenCount;
                stepTokens += batchTokens;
                tokensSeen += batchTokens;
                accumCount++;
                batch.Dispose();

                // Optimizer step (after accumulation)
                if (accumCount < _config.GradAccumSteps) continue;

                float lr = _scheduler.GetLr(step + 1);
                _optimizer.LearningRate = lr;

                // Each microbatch contributed a mean gradient; the optimizer step
                // needs their mean, not their sum, or effective LR is K times too high.
                Gradients.Scale(_parameters, 1f / accumCount);

                float gradNorm = _config.GradClipNorm > 0f
                    ? Gradients.ClipGlobalNorm(_parameters, _config.GradClipNorm)
                    : 0f;

                _optimizer.Update();

                step++;
                float stepLoss = accumLoss / accumCount;
                accumLoss  = 0f;
                accumCount = 0;
                sw.Stop();

                progress?.Report((float)step / _config.TotalSteps);

                // Callbacks
                float? valLoss = null;
                if (_config.EvalInterval > 0 && evaluate is not null && step % _config.EvalInterval == 0)
                {
                    try
                    {
                        var evalRes = evaluate(step);
                        valLoss = evalRes.Loss;
                    }
                    catch
                    {
                        valLoss = null;
                    }
                }

                if (valLoss.HasValue && float.IsFinite(valLoss.Value) && valLoss.Value < bestValLoss)
                {
                    bestValLoss = valLoss.Value;
                    try
                    {
                        string bestDir = Path.Combine(_config.CheckpointDir, "best");
                        Checkpoint.Save(bestDir, _parameters, _optimizer, step, stepLoss, extra: Extra(valLoss));
                        onCheckpoint?.Invoke(bestDir);
                    }
                    catch
                    {
                    }
                }

                if (onStep is not null && _config.LogInterval > 0 && step % _config.LogInterval == 0)
                {
                    onStep(new TrainStepResult
                    {
                        Step            = step,
                        Loss            = stepLoss,
                        LearningRate    = lr,
                        GradNorm        = gradNorm,
                        StepTime        = sw.Elapsed,
                        TokensThisStep  = stepTokens,
                        TokensSeen      = tokensSeen,
                        TokensPerSecond = sw.Elapsed.TotalSeconds > 0 ? (float)(stepTokens / sw.Elapsed.TotalSeconds) : 0f,
                        ValLoss         = valLoss,
                    });
                }
                stepTokens = 0;

                if (_config.CheckpointInterval > 0 && step % _config.CheckpointInterval == 0 && _config.KeepRecent >= 0)
                {
                    string dir = Path.Combine(_config.CheckpointDir, $"step-{step:D07}");
                    Checkpoint.Save(dir, _parameters, _optimizer, step, stepLoss, extra: Extra(valLoss));
                    PruneRetained(dir);
                    onCheckpoint?.Invoke(dir);
                }

                if (step >= _config.TotalSteps) break;
            }
        }
        catch (OperationCanceledException)
        {
            // Persist the latest consistent state so an interrupted run resumes
            // exactly where it stopped, then rethrow for the caller to report.
            if (_config.CheckpointInterval > 0)
            {
                string dir = Path.Combine(_config.CheckpointDir, $"step-{step:D07}");
                Checkpoint.Save(dir, _parameters, _optimizer, step, note: "interrupted", extra: Extra(null));
                onCheckpoint?.Invoke(dir);
            }
            throw;
        }

        // Final checkpoint
        if (_config.CheckpointInterval > 0)
        {
            string dir = Path.Combine(_config.CheckpointDir, $"step-{step:D07}-final");
            Checkpoint.Save(dir, _parameters, _optimizer, step, note: "final", extra: Extra(null));
            onCheckpoint?.Invoke(dir);
        }
    }

    /// <summary>
    /// Deletes the oldest rolling step-* checkpoints when
    /// <see cref="TrainConfig.KeepRecent"/> is non-zero, newest N retained.
    /// The emitted <paramref name="justSaved"/> directory is always kept.
    /// </summary>
    private void PruneRetained(string justSaved)
    {
        int keep = _config.KeepRecent;
        if (keep <= 0 || string.IsNullOrWhiteSpace(_config.CheckpointDir)) return;
        if (!Directory.Exists(_config.CheckpointDir)) return;

        var candidates = Directory.GetDirectories(_config.CheckpointDir, "step-*")
            .Where(d => !d.EndsWith("-final", StringComparison.Ordinal))
            .Where(d => !string.Equals(d, justSaved, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(d => ParseStep(d));

        // `candidates` is newest-first. Keep the newest `keep - 1` others
        // (the just-saved directory is already excluded) and delete the rest.
        int keepOthers = Math.Max(0, keep - 1);
        foreach (var stale in candidates.Skip(keepOthers))
        {
            try { Directory.Delete(stale, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    /// <summary>
    /// Everything an interrupted run carries forward into its next checkpoint.
    /// <c>batchesConsumed</c> is the data cursor (see
    /// <see cref="TrainConfig.ResumeDataCursor"/>); <c>tokensSeen</c> and
    /// <c>bestValLoss</c> keep those figures continuous across resumes;
    /// <c>dataSeed</c> records the RNG the cursor was computed against;
    /// <c>dataFingerprint</c> records which corpus the cursor indexes; and
    /// <c>vocabSize</c> records the embedding width so a strict resume onto a
    /// different vocabulary fails with a clear message (see
    /// <see cref="TrainConfig.ExpectedVocabSize"/>).
    /// </summary>
    private static Dictionary<string, string> BuildExtra(
        long tokensSeen, int batchesConsumed, float bestValLoss, float? valLoss, int vocabSize,
        string? dataFingerprint)
    {
        var extra = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["batchesConsumed"] = batchesConsumed.ToString(System.Globalization.CultureInfo.InvariantCulture),
            ["tokensSeen"] = tokensSeen.ToString(System.Globalization.CultureInfo.InvariantCulture),
            ["vocabSize"] = vocabSize.ToString(System.Globalization.CultureInfo.InvariantCulture),
        };
        if (dataFingerprint is not null)
            extra["dataFingerprint"] = dataFingerprint;
        if (valLoss is { } vl && float.IsFinite(vl))
            extra["valLoss"] = vl.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        if (float.IsFinite(bestValLoss))
            extra["bestValLoss"] = bestValLoss.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        return extra;
    }

    /// <summary>
    /// Records the data seed a checkpoint was produced under, when the run
    /// declares one. Read back by <see cref="GuardDataSeed"/> on resume.
    /// </summary>
    private void WriteDataSeed(Dictionary<string, string> extra)
    {
        if (_config.DataSeed is { } seed)
            extra["dataSeed"] = seed.ToString(System.Globalization.CultureInfo.InvariantCulture);
    }

    /// <summary>
    /// A resume that skips forward in the batch stream only reproduces the
    /// original stream when the batcher's RNG starts where it did before. When
    /// the run declares its data seed, a checkpoint recorded under a different
    /// one is a silently wrong resume - reject it rather than train on a stream
    /// that no longer matches the cursor about to be applied.
    /// </summary>
    private void GuardDataSeed(IReadOnlyDictionary<string, string> extra)
    {
        if (_config.DataSeed is not { } configured) return;
        if (!extra.TryGetValue("dataSeed", out var recorded)) return;
        if (!int.TryParse(recorded, System.Globalization.NumberStyles.Integer,
                System.Globalization.CultureInfo.InvariantCulture, out int recordedSeed))
            return;
        if (recordedSeed == configured) return;

        throw new InvalidOperationException(
            $"Checkpoint under '{_config.ResumeFrom}' records data seed {recordedSeed} but this " +
            $"run is configured with {configured}: the resume cursor would replay a different " +
            "data stream than the one it was computed for. Use the recorded seed, or set " +
            $"{nameof(TrainConfig.DataSeed)} to null to opt out of the check.");
    }

    private static int ReadExtraInt(IReadOnlyDictionary<string, string> extra, string key)
    {
        if (!extra.TryGetValue(key, out var raw) || string.IsNullOrWhiteSpace(raw)) return 0;
        return int.TryParse(raw, System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture, out int v) ? v : 0;
    }

    private static long ReadExtraLong(IReadOnlyDictionary<string, string> extra, string key)
    {
        if (!extra.TryGetValue(key, out var raw) || string.IsNullOrWhiteSpace(raw)) return 0;
        return long.TryParse(raw, System.Globalization.NumberStyles.Integer,
            System.Globalization.CultureInfo.InvariantCulture, out long v) ? v : 0;
    }

    private static float? ReadExtraFloat(IReadOnlyDictionary<string, string> extra, string key)
    {
        if (!extra.TryGetValue(key, out var raw) || string.IsNullOrWhiteSpace(raw)) return null;
        return float.TryParse(raw, System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out float v) ? v : null;
    }

    private static long ParseStep(string dir)
    {
        // "â€¦/step-0000123" or "â€¦/step-0000123-final"
        var name = Path.GetFileName(dir);
        if (!name.StartsWith("step-", StringComparison.Ordinal)) return 0;
        int end = name.IndexOf('-', 5);
        var number = end < 0 ? name[5..] : name[5..end];
        return long.TryParse(number, out var v) ? v : 0;
    }

    // Forward + backward â€” delegated to the engine (CPU reference or an accelerator plugin's).

    private float ForwardBackward(TrainingBatch batch, CancellationToken cancellationToken)
        => _engine.ForwardBackward(batch, cancellationToken);
}
