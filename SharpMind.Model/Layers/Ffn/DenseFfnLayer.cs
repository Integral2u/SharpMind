using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

public sealed class DenseFfnLayer(ModelConfig config, ActivationOps acts, QuantizationOps qOps, TransformerWeights.BlockWeights? weights = null, Dictionary<string, string>? mapping = null) : FfnLayer(config, acts, FfnKind.Dense, qOps, weights, mapping)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x, SharpMind.Core.Memory.Workspace? workspace = null)
        => FfnKernels.Dense(x, W1!, W2!, Acts, workspace);

    public override (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x)
    {
        var hidden = W1!.Forward(x);
        var activated = Acts.Activate(hidden);
        hidden.Dispose();
        var output = W2!.Forward(activated);
        activated.Dispose();
        
        var state = new FfnLayerState { Input = x, Output = output, Kind = FfnKind.Dense };
        return (output, state);
    }

    public override unsafe Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state)
    {
        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);

        using var w2T = W2!.Weight.Transpose();
        var dHidden = new Tensor<float>(gradOutput.Shape.Rows, W2.InFeatures);
        fn(gradOutput.DataPtr, (byte*)w2T.DataPtr, dHidden.DataPtr, gradOutput.Shape.Rows, gradOutput.Shape.Cols, W2.InFeatures);

        using var w1T = W1!.Weight.Transpose();
        var gradInput = new Tensor<float>(dHidden.Shape.Rows, W1.InFeatures);
        fn(dHidden.DataPtr, (byte*)w1T.DataPtr, gradInput.DataPtr, dHidden.Shape.Rows, dHidden.Shape.Cols, W1.InFeatures);

        dHidden.Dispose();
        w2T.Dispose();
        w1T.Dispose();
        return gradInput;
    }
}

