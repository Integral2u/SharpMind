using SharpMind.Model;
using SharpMind.Core.Training;
using SharpMind.Data;
using SharpMind.Training.Schedulers;
using SharpMind.Training.Optimizers;

namespace SharpMind.Training;

/// <summary>
/// High-level trainer that orchestrates the end-to-end training process.
/// </summary>
public sealed class Trainer(
    Transformer model,
    DataLoader loader,
    IOptimizer optimizer,
    IScheduler scheduler,
    ILoss<int> lossFn)
{
    private readonly Transformer _model = model;
    private readonly DataLoader _loader = loader;
    private readonly IOptimizer _optimizer = optimizer;
    private readonly IScheduler _scheduler = scheduler;
    private readonly ILoss<int> _lossFn = lossFn;
    private readonly List<Parameter> _parameters = [.. model.Parameters()];

    /// <summary>
    /// Runs the training loop for a specified number of steps.
    /// </summary>
    public async Task TrainAsync(int totalSteps, CancellationToken ct = default)
    {
        int currentStep = 0;
        float runningLoss = 0;

        Console.WriteLine($"Starting training for {totalSteps} steps...");

        await foreach (var batch in _loader.LoadAsync(ct))
        {
            if (currentStep >= totalSteps) break;

            // 1. Forward Pass
            using var logits = _model.Forward(batch.TokenIds);
            
            // Reshape for cross-entropy: [Batch * Seq, Vocab]
            int batchSize = batch.TokenIds.Shape[0];
            int seqLen = batch.TokenIds.Shape[1];
            using var flatLogits = logits.Reshape(batchSize * seqLen, _model.Config.VocabSize);
            using var flatLabels = batch.Labels.Reshape(batchSize * seqLen);

            // 2. Compute Loss
            float loss = _lossFn.Compute(flatLogits, flatLabels);
            runningLoss = runningLoss * 0.99f + loss * 0.01f;

            // 3. Backward Pass
            using var dLogits = _lossFn.Backward(flatLogits, flatLabels);

            // 4. Update Weights
            _optimizer.Update();

            // 5. Cleanup
            foreach (var p in _parameters) p.ZeroGrad();

            currentStep++;

            if (currentStep % 10 == 0)
            {
                Console.WriteLine($"Step {currentStep}/{totalSteps} | Loss: {runningLoss:F4} | LR: {_scheduler.GetLr(currentStep):F6}");
            }
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
