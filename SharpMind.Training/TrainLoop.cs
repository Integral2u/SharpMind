using SharpMind.Core;
using SharpMind.Core.Tensors;
using SharpMind.Data;
using SharpMind.Data.Batching;
using SharpMind.Model;
using SharpMind.Core.Training;
using SharpMind.Training.Loss;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Autograd;

namespace SharpMind.Training;

/// <summary>
/// The training loop. Connects data → model → loss → backward → optimizer → checkpoint.
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
    private readonly Transformer           _model;
    private readonly List<Parameter>       _parameters;
    private readonly DataLoader            _loader;
    private readonly IOptimizer            _optimizer;
    private readonly IScheduler            _scheduler;
    private readonly TrainConfig           _config;
    private readonly ILoss<int>            _loss;
    private readonly TrainingOps           _ops;
    private readonly GradientMapping       _mapping;
    private readonly BackpropEngine        _engine;

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
        TrainConfig?           config    = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(scheduler);

        _model      = model;
        _parameters = [.. parameters];
        _loader     = loader;
        _optimizer  = optimizer;
        _scheduler  = scheduler;
        _config     = config ?? new TrainConfig();
        _loss       = loss ?? new CrossEntropyLoss(labelSmoothing: _config.LabelSmoothing);
        _mapping    = mapping ?? GradientMappingFactory.Create(smmConfig ?? new SharpMindConfig());
        _ops        = ops;

        if (_config.QuantAwareTraining is { } qat)
            model.EnableQuantAwareTraining(qat);

        _engine     = new BackpropEngine(model, _mapping, _parameters, smmConfig ?? new SharpMindConfig());
    }

    /// <summary>
    /// Runs the training loop for <see cref="TrainConfig.TotalSteps"/> steps.
    /// <paramref name="onStep"/> is called after each optimizer step with loss
    /// and diagnostic info — use for logging, early stopping, or eval triggers.
    /// <paramref name="progress"/> reports 0..1 after each optimizer step.
    /// <paramref name="onCheckpoint"/> is called with the full path of each
    /// saved checkpoint directory (rolling and final), after retention pruning.
    /// </summary>
    public async Task RunAsync(
        Action<TrainStepResult>? onStep            = null,
        IProgress<float>?        progress           = null,
        CancellationToken        cancellationToken  = default,
        Action<string>?          onCheckpoint       = null)
    {
        int startStep = 0;

        if (_config.ResumeFrom is not null)
        {
            var meta = Checkpoint.Load(_config.ResumeFrom, _parameters, _optimizer);
            startStep = meta.Step;
        }

        int   step        = startStep;
        int   accumCount  = 0;
        float accumLoss   = 0f;

        await foreach (var batch in _loader.LoadAsync(cancellationToken))
        {
            cancellationToken.ThrowIfCancellationRequested();

            // Zero gradients at the start of each accumulation window
            if (accumCount == 0) _optimizer.ZeroGrad();

            var sw = System.Diagnostics.Stopwatch.StartNew();

            // Forward pass
            float batchLoss = ForwardBackward(batch);
            accumLoss += batchLoss;
            accumCount++;
            batch.Dispose();

            // Optimizer step (after accumulation)
            if (accumCount < _config.GradAccumSteps) continue;

            float lr = _scheduler.GetLr(step + 1);
            _optimizer.LearningRate = lr;

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
            if (onStep is not null && step % _config.LogInterval == 0)
                onStep(new TrainStepResult
                {
                    Step         = step,
                    Loss         = stepLoss,
                    LearningRate = lr,
                    GradNorm     = gradNorm,
                    StepTime     = sw.Elapsed,
                });

            if (_config.CheckpointInterval > 0 && step % _config.CheckpointInterval == 0 && _config.KeepRecent >= 0)
            {
                string dir = Path.Combine(_config.CheckpointDir, $"step-{step:D07}");
                Checkpoint.Save(dir, _parameters, _optimizer, step, stepLoss);
                PruneRetained(dir);
                onCheckpoint?.Invoke(dir);
            }

            if (step >= _config.TotalSteps) break;
        }

        // Final checkpoint
        if (_config.CheckpointInterval > 0)
        {
            string dir = Path.Combine(_config.CheckpointDir, $"step-{step:D07}-final");
            Checkpoint.Save(dir, _parameters, _optimizer, step, note: "final");
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

        int deleteCount = candidates.Count() - (keep - 1);
        if (deleteCount <= 0) return;

        foreach (var stale in candidates.Take(deleteCount))
        {
            try { Directory.Delete(stale, recursive: true); }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
        }
    }

    private static long ParseStep(string dir)
    {
        // "…/step-0000123" or "…/step-0000123-final"
        var name = Path.GetFileName(dir);
        if (!name.StartsWith("step-", StringComparison.Ordinal)) return 0;
        int end = name.IndexOf('-', 5);
        var number = end < 0 ? name[5..] : name[5..end];
        return long.TryParse(number, out var v) ? v : 0;
    }

    // Forward + backward

    private float ForwardBackward(TrainingBatch batch)
    {
        int batch2  = batch.TokenIds.Shape.Rows;
        int seqLen  = batch.TokenIds.Shape.Cols;
        int vocab   = _model.Config.VocabSize;

        using var ctx       = new ForwardContext();
        using var flatLabels = batch.Labels.Reshape(batch2 * seqLen);

        // Recording forward — ctx owns the returned logits (disposed on scope exit).
        var logits = _engine.ForwardAndRecord(ctx, batch.TokenIds);
        using var logitsFlat = logits.Reshape(batch2 * seqLen, vocab);

        float loss = _loss.Compute(logitsFlat, flatLabels);

        using var dLogits = _loss.Backward(logitsFlat, flatLabels);
        using var flatIds = batch.TokenIds.Reshape(batch2 * seqLen);

        _engine.Backward(ctx, dLogits, flatIds);

        return loss;
    }
}
