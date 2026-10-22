namespace SharpMind.Core.Training;

/// <summary>Name and dimensions of a parameter as stored in a checkpoint.</summary>
public readonly record struct ParameterLayout(string Name, int[] Shape)
{
    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (int d in Shape) n *= d;
            return n;
        }
    }
}

/// <summary>
/// Base interface for a training optimizer.
/// </summary>
public interface IOptimizer : IDisposable
{
    float LearningRate { get; set; }
    int Step { get; }

    /// <summary>
    /// The parameter instances the optimizer updates. Gradients must be
    /// accumulated into these exact instances (not a fresh
    /// <c>model.Parameters()</c> call), otherwise the optimizer reads a
    /// different gradient buffer than the backward pass writes.
    /// </summary>
    IReadOnlyList<Parameter> Parameters { get; }

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

    /// <summary>
    /// Loads optimizer state for a warm restart, where target parameters may be
    /// larger than the checkpoint's (notably a grown embedding table). Optimizers
    /// that cannot map state across shapes fall back to the strict loader.
    /// </summary>
    void LoadStateForWarmRestart(
        BinaryReader reader, int step, IReadOnlyList<ParameterLayout> checkpointParameters)
        => LoadState(reader, step);
}
