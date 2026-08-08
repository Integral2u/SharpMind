using SharpMind.Model;
using SharpMind.Core;
using SharpMind.Core.Training;
using SharpMind.Data;
using SharpMind.Training.Autograd;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Optimizers;

namespace SharpMind.Training;

/// <summary>
/// High-level trainer that orchestrates the end-to-end training process.
/// </summary>
public sealed class Trainer
{
    private readonly Transformer _model;
    private readonly DataLoader _loader;
    private readonly IOptimizer _optimizer;
    private readonly IScheduler _scheduler;
    private readonly ILoss<int> _lossFn;
    private readonly List<Parameter> _parameters;
    private readonly BackpropEngine _engine;

    public Trainer(
        Transformer model,
        DataLoader loader,
        IOptimizer optimizer,
        IScheduler scheduler,
        ILoss<int> lossFn,
        SharpMindConfig? smmConfig = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(loader);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(scheduler);
        ArgumentNullException.ThrowIfNull(lossFn);

        _model = model;
        _loader = loader;
        _optimizer = optimizer;
        _scheduler = scheduler;
        _lossFn = lossFn;
        _parameters = [.. optimizer.Parameters];

        var config = smmConfig ?? new SharpMindConfig();
        var mapping = GradientMappingFactory.Create(config);
        _engine = new BackpropEngine(model, mapping, _parameters, config);
    }

    /// <summary>
    /// Runs the training loop for a specified number of steps.
    /// <paramref name="progress"/> reports 0..1 after every batch step.
    /// </summary>
    public async Task TrainAsync(int totalSteps, IProgress<float>? progress = null, CancellationToken ct = default)
    {
        int currentStep = 0;
        float runningLoss = 0;

        await foreach (var batch in _loader.LoadAsync(ct))
        {
            if (currentStep >= totalSteps) break;

            // 1. Forward Pass (recording, so backprop can run)
            using var ctx = new ForwardContext();
            int batchSize = batch.TokenIds.Shape[0];
            int seqLen = batch.TokenIds.Shape[1];
            using var flatLabels = batch.Labels.Reshape(batchSize * seqLen);
            using var flatIds = batch.TokenIds.Reshape(batchSize * seqLen);

            // BackpropEngine.ForwardAndRecord returns flat [Batch*Seq, Vocab] logits.
            using var logits = _engine.ForwardAndRecord(ctx, batch.TokenIds);

            // 2. Compute Loss
            float loss = _lossFn.Compute(logits, flatLabels);
            runningLoss = runningLoss * 0.99f + loss * 0.01f;

            // 3. Backward Pass
            using var dLogits = _lossFn.Backward(logits, flatLabels);
            _engine.Backward(ctx, dLogits, flatIds);

            // 4. Update Weights
            _optimizer.LearningRate = _scheduler.GetLr(currentStep + 1);
            _optimizer.Update();

            // 5. Cleanup
            foreach (var p in _parameters) p.ZeroGrad();

            currentStep++;
            progress?.Report((float)currentStep / totalSteps);
        }
    }

    /// <summary>
    /// Saves the current model and optimizer state to a checkpoint file.
    /// </summary>
    public void SaveCheckpoint(string path)
    {
        using var fs = new FileStream(path, FileMode.Create);
        using var writer = new BinaryWriter(fs);

        // Save ModelConfig (simplified)
        writer.Write(_model.Config.HiddenDim);
        writer.Write(_model.Config.NumLayers);
        writer.Write(_model.Config.VocabSize);

        // Save Parameters
        foreach (var p in _parameters)
        {
            writer.Write(p.Name);
            writer.Write(p.Data.Data.Length);
            foreach (var val in p.Data.Data) writer.Write(val);
        }

        // Save Optimizer State
        if (_optimizer is AdamW adam)
        {
            adam.SaveState(writer);
        }
    }

    public void LoadCheckpoint(string path)
    {
        using var fs = new FileStream(path, FileMode.Open);
        using var reader = new BinaryReader(fs);

        int hidden = reader.ReadInt32();
        int layers = reader.ReadInt32();
        int vocab = reader.ReadInt32();

        foreach (var p in _parameters)
        {
            string name = reader.ReadString();
            int len = reader.ReadInt32();
            for (int i = 0; i < len; i++) p.Data.Data[i] = reader.ReadSingle();
        }

        if (_optimizer is AdamW adam)
        {
            adam.LoadState(reader, 0); // Step is handled by optimizer internally or passed here
        }
    }
}