using System.Diagnostics;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Data.Batching;
using SharpMind.Model;
using SharpMind.Training.Loss;

namespace SharpMind.Training;

/// <summary>
/// Trains every parameter for next-token prediction using finite-difference
/// gradients (central differences, ±h) applied by an <see cref="IOptimizer"/>.
///
/// Unlike <see cref="TrainLoop"/>, this never runs a backward pass — each
/// gradient element is estimated by two extra forward passes. That makes it
/// ideal for tiny teaching/toy models (O(parameters) forwards per step) but
/// unusable for real-scale LLMs; prefer backprop for anything non-trivial.
///
/// The trainer consumes <see cref="TrainingBatch"/> streams produced by the
/// data pipeline (e.g. <c>loader.LoadAsync()</c> or the LearnableGenerator
/// adapter) so pseudo-language and real data share one code path.
/// </summary>
public sealed class FiniteDifferenceTrainer
{
    private readonly Transformer _model;
    private readonly IAsyncEnumerable<TrainingBatch> _batches;
    private readonly IOptimizer _optimizer;
    private readonly ILoss<int> _loss;
    private readonly FiniteDifferenceConfig _config;
    private readonly List<Parameter> _parameters;

    /// <param name="model">Training transformer whose parameters are optimised.</param>
    /// <param name="batches">Stream of training batches. The loop consumes one batch per step
    /// and stops after <see cref="FiniteDifferenceConfig.TotalSteps"/> steps.</param>
    /// <param name="optimizer">Applied after each batch (e.g. <c>AdamW</c>).</param>
    /// <param name="parameters">
    /// The parameter instances to optimise. Must be the <em>same</em> instances
    /// the <paramref name="optimizer"/> was constructed from, otherwise the
    /// optimizer would read a different (always-zero) gradient buffer — each
    /// <c>model.Parameters()</c> call allocates fresh <c>Parameter.Grad</c>
    /// tensors. Capture the list once and share it. Defaults to
    /// <c>model.Parameters()</c>.
    /// </param>
    /// <param name="loss">Loss function; defaults to <see cref="CrossEntropyLoss"/>.</param>
    /// <param name="config">Step count, perturbation size, logging and checkpointing.</param>
    public FiniteDifferenceTrainer(
        Transformer model,
        IAsyncEnumerable<TrainingBatch> batches,
        IOptimizer optimizer,
        IEnumerable<Parameter>? parameters = null,
        ILoss<int>? loss = null,
        FiniteDifferenceConfig? config = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(batches);
        ArgumentNullException.ThrowIfNull(optimizer);

        _model = model;
        _batches = batches;
        _optimizer = optimizer;
        _loss = loss ?? new CrossEntropyLoss();
        _config = config ?? new FiniteDifferenceConfig();
        _parameters = parameters is null ? [.. model.Parameters()] : [.. parameters];
    }

    /// <summary>
    /// Runs the training loop for <see cref="FiniteDifferenceConfig.TotalSteps"/>
    /// steps. <paramref name="progress"/> reports 0..1 after every step and
    /// <paramref name="onStep"/> is invoked every
    /// <see cref="FiniteDifferenceConfig.LogInterval"/> steps with step-level
    /// diagnostics.
    /// </summary>
    public async Task<FiniteDifferenceResult> TrainAsync(
        IProgress<float>? progress = null,
        Action<TrainStepResult>? onStep = null,
        CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        int step = 0;
        float finalLoss = float.NaN;
        string? lastCheckpoint = null;

        int batchSize = -1;
        int seqLen = -1;

        await foreach (var batch in _batches.WithCancellation(ct))
        {
            if (step >= _config.TotalSteps)
            {
                batch.Dispose();
                break;
            }

            try
            {
                batchSize = batch.TokenIds.Shape.Rows;
                seqLen = batch.TokenIds.Shape.Cols;

                // Keep the batch's tensors alive for the finite-difference loop.
                _currentTokens = batch.TokenIds;
                _currentLabels = batch.Labels;

                using var logits = _model.Forward(batch.TokenIds);
                using var flatLogits = logits.Reshape(batchSize * seqLen, _model.Config.VocabSize);
                using var flatLabels = batch.Labels.Reshape(batchSize * seqLen);
                finalLoss = _loss.Compute(flatLogits, flatLabels);

                if (!float.IsFinite(finalLoss))
                    throw new InvalidOperationException($"Training loss diverged at step {step}: loss={finalLoss}.");

                _optimizer.ZeroGrad();
                EstimateGradients(batchSize, seqLen, ct);

                _optimizer.Update();
                step++;

                // Callbacks
                progress?.Report((float)step / _config.TotalSteps);
                if (onStep is not null && step % _config.LogInterval == 0)
                {
                    onStep(new TrainStepResult
                    {
                        Step = step,
                        Loss = finalLoss,
                        LearningRate = _optimizer.LearningRate,
                        GradNorm = ComputeGradNorm(),
                        StepTime = sw.Elapsed,
                    });
                }

                if (_config.CheckpointInterval > 0 && step % _config.CheckpointInterval == 0)
                {
                    lastCheckpoint = Path.Combine(_config.CheckpointDir, $"step-{step:D07}");
                    Checkpoint.Save(lastCheckpoint, _parameters, _optimizer, step, finalLoss);
                }
            }
            finally
            {
                _currentTokens = null;
                _currentLabels = null;
                batch.Dispose();
            }
        }

        sw.Stop();
        return new FiniteDifferenceResult
        {
            FinalLoss = finalLoss,
            Steps = step,
            Elapsed = sw.Elapsed,
            CheckpointPath = lastCheckpoint,
        };
    }

    /// <summary>
    /// Estimates the gradient of the current loss with respect to every
    /// parameter element using central differences:
    ///   dL/dpᵢ ≈ (L(p + h·eᵢ) − L(p − h·eᵢ)) / 2h
    /// Writes the estimate into <see cref="Parameter.Grad"/> (overwrite, not accumulate).
    /// </summary>
    private void EstimateGradients(int batchSize, int seqLen, CancellationToken ct)
    {
        int vocab = _model.Config.VocabSize;
        float h = _config.Perturbation;

        foreach (var p in _parameters)
        {
            var data = p.Data.Data;
            var grad = p.Grad.Data;
            for (int i = 0; i < data.Length; i++)
            {
                float original = data[i];
                data[i] = original + h;
                float plus = LossFor(batchSize, seqLen, vocab);
                data[i] = original - h;
                float minus = LossFor(batchSize, seqLen, vocab);
                data[i] = original;
                grad[i] = (plus - minus) / (2 * h);
            }
        }
    }

    private float LossFor(int batchSize, int seqLen, int vocab)
    {
        // The current batch's token tensor is passed straight through the
        // streamed batch; it stays alive for the duration of the step.
        // Re-computing the forward here re-reads the just-perturbed weights.
        using var logits = _model.Forward(_currentTokens!);
        using var flatLogits = logits.Reshape(batchSize * seqLen, vocab);
        using var flatLabels = _currentLabels!.Reshape(batchSize * seqLen);
        return _loss.Compute(flatLogits, flatLabels);
    }

    private float ComputeGradNorm()
    {
        double sum = 0;
        foreach (var p in _parameters)
        {
            var grad = p.Grad.Data;
            for (int i = 0; i < grad.Length; i++)
                sum += (double)grad[i] * grad[i];
        }
        return (float)Math.Sqrt(sum);
    }

    private Tensor<int>? _currentTokens;
    private Tensor<int>? _currentLabels;
}
