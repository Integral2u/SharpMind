using SharpMind.Core;
using SharpMind.Core.Training;
using SharpMind.Data.Batching;
using SharpMind.Model;
using SharpMind.Training.Autograd;

namespace SharpMind.Training;

/// <summary>
/// The reference <see cref="ITrainingEngine"/>: <see cref="BackpropEngine"/> plus
/// an <see cref="ILoss{TLabel}"/>, exactly the step <see cref="TrainLoop"/> ran
/// before engines were pluggable. What every accelerator engine is measured against.
/// </summary>
public sealed class CpuTrainingEngine : ITrainingEngine
{
    private readonly Transformer    _model;
    private readonly ILoss<int>     _loss;
    private readonly BackpropEngine _engine;

    public CpuTrainingEngine(Transformer model, GradientMapping mapping, IReadOnlyList<Parameter> parameters, SharpMindConfig config, ILoss<int> loss)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(mapping);
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(loss);

        _model  = model;
        _loss   = loss;
        _engine = new BackpropEngine(model, mapping, parameters, config);
    }

    /// <inheritdoc />
    public float ForwardBackward(TrainingBatch batch, CancellationToken cancellationToken = default)
    {
        int batch2 = batch.TokenIds.Shape.Rows;
        int seqLen = batch.TokenIds.Shape.Cols;
        int vocab  = _model.Config.VocabSize;

        using var ctx        = new ForwardContext();
        using var flatLabels = batch.Labels.Reshape(batch2 * seqLen);

        // Recording forward — ctx owns the returned logits (disposed on scope exit).
        var logits = _engine.ForwardAndRecord(ctx, batch.TokenIds, cancellationToken);
        using var logitsFlat = logits.Reshape(batch2 * seqLen, vocab);

        float loss = _loss.Compute(logitsFlat, flatLabels);

        using var dLogits = _loss.Backward(logitsFlat, flatLabels);
        using var flatIds = batch.TokenIds.Reshape(batch2 * seqLen);

        _engine.Backward(ctx, dLogits, flatIds, cancellationToken);

        return loss;
    }

    public void Dispose() => _engine.Dispose();
}
