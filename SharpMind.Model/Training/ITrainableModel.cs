using SharpMind.Core.Tensors;
using SharpMind.Core.Training;

namespace SharpMind.Model.Training;

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
