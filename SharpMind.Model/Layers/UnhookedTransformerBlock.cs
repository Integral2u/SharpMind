using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Model.Layers.Attention;
using SharpMind.Model.Layers.Ffn;
using SharpMind.Model.Layers.ShortConv;

namespace SharpMind.Model.Layers;

public sealed class UnhookedTransformerBlock(int layerIdx, AttentionLayer? attention, FfnLayer ffn, NormLayer norm1, NormLayer norm2,
    NormLayer? postAttnNorm = null, NormLayer? postFfnNorm = null, ShortConvLayer? shortConv = null, bool postNormResidual = false)
    : TransformerBlock(layerIdx, attention, ffn, norm1, norm2, postAttnNorm, postFfnNorm, shortConv, postNormResidual)
{
    public override Tensor<float> Forward(Tensor<float> x, IKVCache? cache, int positionOffset = 0, bool causal = true, IWorkspace? workspace = null, int windowSize = 0)
    {
        ThrowIfDisposed();

        if (_postNormResidual)
        {
            // BERT post-norm: no pre-norm; normalise after each residual add, and
            // the FFN reads the post-attention-normed state.
            var postAttnOut = FirstBranch(x, cache, positionOffset, causal, workspace, windowSize);
            x.AddInPlace(postAttnOut);
            postAttnOut.Dispose();
            _norm1.ForwardInPlace(x);

            var postFfnOut = _ffn.Forward(x, workspace);
            x.AddInPlace(postFfnOut);
            postFfnOut.Dispose();
            _norm2.ForwardInPlace(x);

            return x;
        }

        var normed1 = _norm1.Forward(x, workspace);

        var attnOut = FirstBranch(normed1, cache, positionOffset, causal, workspace, windowSize);
        normed1.Dispose();

        if (_postAttnNorm != null)
        {
            var postNormed = _postAttnNorm.Forward(attnOut, workspace);
            attnOut.Dispose();
            attnOut = postNormed;
        }

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

        x.AddInPlace(ffnOut);
        ffnOut.Dispose();

        return x;
    }
}
