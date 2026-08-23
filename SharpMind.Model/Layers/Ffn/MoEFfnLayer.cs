using SharpMind.Core;
using SharpMind.Core.Activations;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

public sealed class MoEFfnLayer(ModelConfig config, ActivationOps acts, QuantizationOps qOps, TransformerWeights.BlockWeights? weights = null, Dictionary<string, string>? mapping = null) : FfnLayer(config, acts, FfnKind.MoE, qOps, weights, mapping)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x, SharpMind.Core.Memory.IWorkspace? workspace = null)
        => FfnKernels.MoE(x, Router!, ExpertGate!, ExpertUp!, ExpertDown!, Config.TopKExperts, Acts, workspace);

    public override (Tensor<float> Output, FfnLayerState State) ForwardWithState(Tensor<float> x)
    {
        var output = ApplyFfn(x);
        var state = new FfnLayerState { Input = x, Output = output, Kind = FfnKind.MoE };
        return (output, state);
    }

    public override Tensor<float> Backward(Tensor<float> gradOutput, FfnLayerState state) => gradOutput; // Stub - MoE backward is complex
}

