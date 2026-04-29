using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

/// <summary>Concrete gated FFN — routes ApplyFfn to the JigSaw-assembled kernel.</summary>
public sealed class GatedFfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops) : FfnLayer(config, acts, ops, FfnKind.Gated)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x)
        => FfnKernels.Gated(x, WGate!, WUp!, WDown!, Acts, Ops);
}

