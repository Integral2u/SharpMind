namespace SharpMind.Training.Schedulers;

/// <summary>
/// Cosine decay with linear warmup — the standard LLM training schedule.
///
/// Three phases:
///   1. Linear warmup [0, warmupSteps]: lr grows from 0 to maxLr.
///   2. Cosine decay  [warmupSteps, decaySteps]: lr decays from maxLr to minLr.
///   3. Constant      [decaySteps, ∞]: lr stays at minLr.
///
/// Used by GPT-3, LLaMA 2, Mistral, and most modern open-weight models.
/// </summary>
public sealed class CosineWithWarmup : IScheduler
{
    private readonly float _maxLr;
    private readonly float _minLr;
    private readonly int   _warmupSteps;
    private readonly int   _decaySteps;

    /// <param name="maxLr">Peak learning rate after warmup. Typically 1e-4 to 3e-4.</param>
    /// <param name="minLr">
    /// Minimum LR at end of decay. Typically maxLr/10.
    /// LLaMA 2 uses maxLr * 0.1.
    /// </param>
    /// <param name="warmupSteps">Steps for linear warmup. Typically 2000 for large runs.</param>
    /// <param name="decaySteps">
    /// Total steps including warmup. Set to total training steps.
    /// After this the LR stays at minLr.
    /// </param>
    public CosineWithWarmup(float maxLr, float minLr, int warmupSteps, int decaySteps)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxLr);
        ArgumentOutOfRangeException.ThrowIfNegative(minLr);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(warmupSteps);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(decaySteps);
        if (decaySteps < warmupSteps)
            throw new ArgumentException("decaySteps must be >= warmupSteps.");

        _maxLr       = maxLr;
        _minLr       = minLr;
        _warmupSteps = warmupSteps;
        _decaySteps  = decaySteps;
    }

    public float GetLr(int step)
    {
        if (step < _warmupSteps)
            return _maxLr * step / _warmupSteps;

        if (step >= _decaySteps)
            return _minLr;

        float progress = (float)(step - _warmupSteps) / (_decaySteps - _warmupSteps);
        float cosine   = 0.5f * (1f + MathF.Cos(MathF.PI * progress));
        return _minLr + cosine * (_maxLr - _minLr);
    }
}
