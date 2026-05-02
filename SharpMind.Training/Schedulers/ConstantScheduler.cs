namespace SharpMind.Training.Schedulers;

/// <summary>
/// Constant learning rate. No warmup or decay.
/// Useful for fine-tuning with a pre-warmed optimizer state.
/// </summary>
public sealed class ConstantScheduler(float lr) : IScheduler
{
    private readonly float _lr = lr;

    public float GetLr(int step) => _lr;
}
