using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

public sealed class DenseFfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops, QuantizationOps qOps, TransformerWeights.BlockWeights? weights = null) : FfnLayer(config, acts, ops, FfnKind.Dense, qOps, weights)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x, SharpMind.Core.Memory.Workspace? workspace = null)
        => FfnKernels.Dense(x, W1!, W2!, Acts, Ops, workspace);

    public override (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x)
    {
        var hidden = W1!.Forward(x, Ops);
        var activated = Acts.Activate(hidden);
        hidden.Dispose();
        var output = W2!.Forward(activated, Ops);
        activated.Dispose();
        
        var state = new FfnLayerState { Input = x, Output = output, Kind = FfnKind.Dense };
        return (output, state);
    }

    public override Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state)
    {
        // Simplified: just propagate gradient backward
        // Real implementation needs SiLU derivative
        using var w2T = TensorOps.Transpose(W2!.Weight);
        var dHidden = Ops.MatMul(gradOutput, w2T);
        using var w1T = TensorOps.Transpose(W1!.Weight);
        var gradInput = Ops.MatMul(dHidden, w1T);
        dHidden.Dispose();
        w2T.Dispose();
        w1T.Dispose();
        return gradInput;
    }
}

