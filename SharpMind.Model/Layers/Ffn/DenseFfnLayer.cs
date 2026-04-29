using SharpMind.Core.Activations;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Ffn;

/// <summary>Concrete dense FFN — routes ApplyFfn to the JigSaw-assembled kernel.</summary>
public sealed class DenseFfnLayer(ModelConfig config, ActivationOps acts, TensorOps ops) : FfnLayer(config, acts, ops, FfnKind.Dense)
{
    public override Tensor<float> ApplyFfn(Tensor<float> x)
        => FfnKernels.Dense(x, W1!, W2!, Acts, Ops);
}

