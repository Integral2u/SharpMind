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
    

    public TrainLoop(
        Transformer           model,
        IEnumerable<Parameter> parameters,
        DataLoader             loader,
        IOptimizer             optimizer,
        IScheduler             scheduler,
        TrainingOps            ops,
        ILoss<int>?            loss     = null,
        GradientMapping?       mapping  = null,        
        TrainConfig?           config   = null)
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
        _loss       = loss ?? new CrossEntropyLoss();
        _mapping    = mapping ?? new GradientMapping();
        _ops        = ops;
    }

    // ── Main loop ─────────────────────────────────────────────────────────

    /// <summary>
    /// Runs the training loop for <see cref="TrainConfig.TotalSteps"/> steps.
    /// <paramref name="onStep"/> is called after each optimizer step with loss
    /// and diagnostic info — use for logging, early stopping, or eval triggers.
    /// </summary>
    public async Task RunAsync(
        Action<TrainStepResult>? onStep            = null,
        CancellationToken        cancellationToken  = default)
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

            // ── Forward pass ──────────────────────────────────────────────
            float batchLoss = ForwardBackward(batch);
            accumLoss += batchLoss;
            accumCount++;
            batch.Dispose();

            // ── Optimizer step (after accumulation) ───────────────────────
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

            // ── Callbacks ─────────────────────────────────────────────────
            if (onStep is not null && step % _config.LogInterval == 0)
                onStep(new TrainStepResult
                {
                    Step         = step,
                    Loss         = stepLoss,
                    LearningRate = lr,
                    GradNorm     = gradNorm,
                    StepTime     = sw.Elapsed,
                });

            if (_config.CheckpointInterval > 0 && step % _config.CheckpointInterval == 0)
            {
                string dir = Path.Combine(_config.CheckpointDir, $"step-{step:D07}");
                Checkpoint.Save(dir, _parameters, _optimizer, step, stepLoss);
            }

            if (step >= _config.TotalSteps) break;
        }

        // Final checkpoint
        if (_config.CheckpointInterval > 0)
        {
            string dir = Path.Combine(_config.CheckpointDir, $"step-{step:D07}-final");
            Checkpoint.Save(dir, _parameters, _optimizer, step, note: "final");
        }
    }

    // ── Forward + backward ────────────────────────────────────────────────

    private float ForwardBackward(TrainingBatch batch)
    {
        int batch2  = batch.TokenIds.Shape.Rows;
        int seqLen  = batch.TokenIds.Shape.Cols;

        using var flatIds    = batch.TokenIds.Reshape(batch2 * seqLen);
        using var flatLabels = batch.Labels.Reshape(batch2 * seqLen);

        using var logits2d = _model.Forward(batch.TokenIds);
        using var logitsFlat = logits2d.Reshape(batch2 * seqLen, _model.Config.VocabSize);

        float loss = _loss.Compute(logitsFlat, flatLabels);

        using var dLogits = _loss.Backward(logitsFlat, flatLabels);

        BackwardEmbedding(dLogits, flatIds);

        return loss;
    }

    private void BackwardEmbedding(Tensor<float> dLogits, Tensor<int> tokenIds)
    {
        var embeddingParam = _parameters.FirstOrDefault(
            p => p.Name.Contains("embedding", StringComparison.OrdinalIgnoreCase));

        if (embeddingParam is not null)
            _mapping.Embedding.Compute(dLogits, tokenIds, embeddingParam);
    }
}
