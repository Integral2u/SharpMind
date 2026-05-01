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

/// <summary>
/// A model that can be trained via backpropagation.
/// </summary>
public interface ITrainableModel
{
    /// <summary>
    /// Performs a forward pass, returning the result and a collection of states for each layer.
    /// </summary>
    (Tensor<float> Output, List<object> States) ForwardTrainable(Tensor<int> tokenIds);

    /// <summary>
    /// Performs a backward pass, propagating gradients through the model and updating parameter gradients.
    /// </summary>
    Tensor<float> Backward(Tensor<float> gradOutput, List<object> states, GradientMapping mapping);
}
