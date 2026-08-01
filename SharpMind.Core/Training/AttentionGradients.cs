using SharpMind.Core.Tensors;

namespace SharpMind.Core.Training;

/// <summary>
/// Gradients produced by scaled dot-product attention backward for one head.
///
/// Returned as a struct (not a tuple) because JigSawDotNet re-emits the method
/// attributes of the assembled slots and cannot round-trip the compiler's
/// <c>TupleElementNames</c> attribute on tuple-typed returns.
/// </summary>
public readonly struct AttentionGradients
{
    public AttentionGradients(Tensor<float> dQ, Tensor<float> dK, Tensor<float> dV)
    {
        DQ = dQ;
        DK = dK;
        DV = dV;
    }

    public Tensor<float> DQ { get; }
    public Tensor<float> DK { get; }
    public Tensor<float> DV { get; }
}
