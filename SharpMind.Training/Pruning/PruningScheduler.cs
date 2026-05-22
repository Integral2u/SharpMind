using SharpMind.Core.Tensors;

namespace SharpMind.Training.Pruning;

/// <summary>
/// Pruning scheduler for gradual increase.
/// </summary>
public class PruningScheduler(float targetSparsity, int totalSteps)
{
    private float _currentSparsity = 0f;
    private readonly float _targetSparsity = targetSparsity;
    private readonly int _totalSteps = totalSteps;
    private int _currentStep;

    public float CurrentSparsity => _currentSparsity;

    public void Step()
    {
        if (_currentStep >= _totalSteps)
        {
            _currentSparsity = _targetSparsity;
            return;
        }

        _currentSparsity = _targetSparsity * ((float)_currentStep / _totalSteps);
        _currentStep++;
    }

    public void Apply(Tensor<float> weights) => PruningKernels.MagnitudePrune(weights, _currentSparsity);
}