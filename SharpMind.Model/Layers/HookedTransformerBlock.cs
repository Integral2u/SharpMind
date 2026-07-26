using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;

namespace SharpMind.Model.Layers;

public sealed class HookedTransformerBlock : TransformerBlock
{
    private IActivationHook? _hook;

    public IActivationHook? Hook => _hook;

    public HookedTransformerBlock(int layerIdx, AttentionLayer attention, FfnLayer ffn, NormLayer norm1, NormLayer norm2,
        NormLayer? postAttnNorm = null, NormLayer? postFfnNorm = null)
        : base(layerIdx, attention, ffn, norm1, norm2, postAttnNorm, postFfnNorm) { }

    public override void SetActivationHook(IActivationHook? hook) => _hook = hook;

    public override Tensor<float> Forward(Tensor<float> x, IKVCache? cache, int positionOffset = 0, bool causal = true, Workspace? workspace = null)
    {
        ThrowIfDisposed();

        var normed1 = _norm1.Forward(x, workspace);
        _hook?.OnPreAttention(_layerIdx, normed1);
        var attnOut = _attention.Forward(normed1, positionOffset, causal, cache, workspace);
        normed1.Dispose();

        if (_postAttnNorm != null)
        {
            var postNormed = _postAttnNorm.Forward(attnOut, workspace);
            attnOut.Dispose();
            attnOut = postNormed;
        }

        _hook?.OnPostAttention(_layerIdx, attnOut);
        x.AddInPlace(attnOut);
        attnOut.Dispose();

        var normed2 = _norm2.Forward(x, workspace);
        var ffnOut = _ffn.Forward(normed2, workspace);
        normed2.Dispose();

        if (_postFfnNorm != null)
        {
            var postNormed = _postFfnNorm.Forward(ffnOut, workspace);
            ffnOut.Dispose();
            ffnOut = postNormed;
        }

        _hook?.OnPostFFN(_layerIdx, ffnOut);
        x.AddInPlace(ffnOut);
        ffnOut.Dispose();

        return x;
    }
}
