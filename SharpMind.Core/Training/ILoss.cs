using SharpMind.Core.Tensors;

namespace SharpMind.Core.Training;

/// <summary>
/// Base interface for a loss function.
/// </summary>
public interface ILoss<TLabel> where TLabel : unmanaged, System.Numerics.INumber<TLabel>
{
    float Compute(Tensor<float> predictions, Tensor<TLabel> labels);
    Tensor<float> Backward(Tensor<float> predictions, Tensor<TLabel> labels);
}
