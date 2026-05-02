namespace SharpMind.Training.Schedulers;

/// <summary>
/// Linear warmup only — holds at maxLr after warmup completes.
/// </summary>
public sealed class LinearWarmup(float maxLr, int warmupSteps) : IScheduler
{
    private readonly float _maxLr = maxLr;
    private readonly int _warmupSteps = warmupSteps;

    public float GetLr(int step)
        => step < _warmupSteps ? _maxLr * step / _warmupSteps : _maxLr;
}
