using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Training;

/// <summary>
/// A layer that can perform both forward and backward passes.
/// </summary>
public interface ITrainableLayer
{
    /// <summary>
    /// Performs the forward pass and returns the result and the state needed for backward.
    /// </summary>
    (Tensor<float> Output, object State) ForwardTrainable(Tensor<float> input, object? extra = null);

    /// <summary>
    /// Performs the backward pass using the saved state and returns the gradient with respect to the input.
    /// </summary>
    Tensor<float> Backward(Tensor<float> gradOutput, object state, GradientMapping mapping, object? extra = null);
}
