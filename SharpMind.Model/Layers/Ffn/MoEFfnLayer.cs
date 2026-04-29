using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

/// <summary>Concrete MoE FFN — routes ApplyFfn to the JigSaw-assembled kernel.</summary>
public sealed class MoEFfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops) : FfnLayer(config, acts, ops, FfnKind.MoE)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x)
        => FfnKernels.MoE(x, Router!, ExpertGate!, ExpertUp!, ExpertDown!,
                          Config.TopKExperts, Acts, Ops);
}

