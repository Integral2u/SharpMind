using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

public sealed class GatedFfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops) : FfnLayer(config, acts, ops, FfnKind.Gated)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x)
        => FfnKernels.Gated(x, WGate!, WUp!, WDown!, Acts, Ops);

    public override (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x)
    {
        var gate = WGate!.Forward(x, Ops);
        var up = WUp!.Forward(x, Ops);
        var activated = Acts.GatedActivate(gate, up);
        gate.Dispose();
        up.Dispose();
        var output = WDown!.Forward(activated, Ops);
        activated.Dispose();
        
        var state = new FfnLayerState { Input = x, Output = output, Kind = FfnKind.Gated };
        return (output, state);
    }

    public override Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state)
    {
        // Simplified gradient propagation
        using var wDownT = TensorOps.Transpose(WDown!.Weight);
        var dHidden = Ops.MatMul(gradOutput, wDownT);
        using var wGateT = TensorOps.Transpose(WGate!.Weight);
        var dGate = Acts.GatedActivate(dHidden, dHidden); // Self gradient for gating
        var gradInput = Ops.MatMul(dGate, wGateT);
        dGate.Dispose();
        wDownT.Dispose();
        wGateT.Dispose();
        return gradInput;
    }
}

