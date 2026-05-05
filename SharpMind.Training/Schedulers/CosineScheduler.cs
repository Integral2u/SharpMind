namespace SharpMind.Training.Schedulers;

/// <summary>
/// Implements a cosine annealing learning rate scheduler.
/// Gradually decreases the learning rate from a peak value to a minimum value
/// following a cosine curve.
/// </summary>
public sealed class CosineScheduler : IScheduler
{
    private readonly float _maxLr;
    private readonly float _minLr;
    private readonly int _totalSteps;

    public CosineScheduler(float maxLr, float minLr = 0f, int totalSteps = 1000)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maxLr);
        ArgumentOutOfRangeException.ThrowIfNegative(minLr);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(totalSteps);

        _maxLr = maxLr;
        _minLr = minLr;
        _totalSteps = totalSteps;
    }

    public float GetLr(int step)
    {
        if (step >= _totalSteps) return _minLr;

        // Cosine decay formula: min + 0.5 * (max - min) * (1 + cos(pi * step / total))
        float ratio = (float)step / _totalSteps;
        return _minLr + 0.5f * (_maxLr - _minLr) * (1f + MathF.Cos(MathF.PI * ratio));
    }
}
