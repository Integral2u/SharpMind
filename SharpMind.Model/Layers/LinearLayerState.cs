using SharpMind.Core.Tensors;

namespace SharpMind.Model.Layers;

public sealed class LinearLayerState
{
    public Tensor<float> Input { get; }
    public int[] InputDims { get; }
    public bool NeedReshape { get; }
    public Tensor<float> WeightGrad { get; }
    public Tensor<float>? BiasGrad { get; set; }

    public LinearLayerState(Tensor<float> originalInput, Tensor<float> flatInput, bool needReshape, Tensor<float> weight)
    {
        Input = flatInput;
        InputDims = originalInput.Shape.Dims.ToArray();
        NeedReshape = needReshape;
        var dims = weight.Shape.Dims.ToArray();
        WeightGrad = Tensor<float>.Zeros(dims);
    }
}
