using SharpMind.Core.Tensors;

namespace SharpMind.Core.Training;

/// <summary>
/// Base interface for a training optimizer.
/// </summary>
public interface IOptimizer : IDisposable
{
    float LearningRate { get; set; }
    int Step { get; }
    void Update();
    void ZeroGrad();

    /// <summary>
    /// Saves optimizer state (moments, step count, etc.) for checkpointing.
    /// </summary>
    void SaveState(BinaryWriter writer);

    /// <summary>
    /// Loads optimizer state from a checkpoint.
    /// </summary>
    void LoadState(BinaryReader reader, int step);
}

/// <summary>
/// Base interface for a loss function.
/// </summary>
public interface ILoss<TLabel> where TLabel : unmanaged, System.Numerics.INumber<TLabel>
{
    float Compute(Tensor<float> predictions, Tensor<TLabel> labels);
    Tensor<float> Backward(Tensor<float> predictions, Tensor<TLabel> labels);
}

/// <summary>
/// Base interface for a gradient computation kernel.
/// Implementation should be a PuzzlePiece.
/// </summary>
public interface IGradientKernel { }
