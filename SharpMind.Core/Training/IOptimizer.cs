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
