using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

public sealed class MoEFfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops) : FfnLayer(config, acts, ops, FfnKind.MoE)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x)
        => FfnKernels.MoE(x, Router!, ExpertGate!, ExpertUp!, ExpertDown!, Config.TopKExperts, Acts, Ops);

    public override (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x)
    {
        var output = ApplyFfn(x);
        var state = new FfnLayerState { Input = x, Output = output, Kind = FfnKind.MoE };
        return (output, state);
    }

    public override Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state)
    {
        return gradOutput; // Stub - MoE backward is complex
    }
}

