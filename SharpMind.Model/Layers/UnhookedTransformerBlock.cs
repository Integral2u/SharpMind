using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.Model.Layers;

public sealed class UnhookedTransformerBlock(int layerIdx, AttentionLayer attention, FfnLayer ffn, NormLayer norm1, NormLayer norm2) : TransformerBlock(layerIdx, attention, ffn, norm1, norm2)
{
    public override Tensor<float> Forward(Tensor<float> x, IKVCache? cache, int positionOffset = 0, bool causal = true, Workspace? workspace = null)
    {
        ThrowIfDisposed();

        var normed1 = _norm1.Forward(x, workspace);

        var attnOut = _attention.Forward(normed1, positionOffset, causal, cache, workspace);
        normed1.Dispose();
        x.AddInPlace(attnOut);
        attnOut.Dispose();

        var normed2 = _norm2.Forward(x, workspace);

        var ffnOut = _ffn.Forward(normed2, workspace);
        normed2.Dispose();
        x.AddInPlace(ffnOut);
        ffnOut.Dispose();

        return x;
    }
}
