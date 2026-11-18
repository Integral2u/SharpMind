using SharpMind.Core.Memory;
using SharpMind.Core.Tensors;
using SharpMind.Model.Layers;

namespace SharpMind.Model;

/// <summary>
/// The multi-token-prediction draft head of a DeepSeek-V3-style / Qwen3.5 NextN
/// model. One extra decoder block that predicts the token after the next one,
/// used to draft tokens a speculative loop can verify with the trunk.
///
/// Per llama.cpp's qwen35/deepseek2 MTP graph, one draft step is:
/// <code>
///   h_norm = RMSNorm(h_trunk, nextn.hnorm)      // trunk's final-normed hidden
///   e_norm = RMSNorm(embed(token), nextn.enorm) // the token being extended
///   x      = nextn.eh_proj([e_norm | h_norm])
///   x      = attention_block(x, position)       // the MTP block's own KV cache
///   out    = RMSNorm(x, nextn.shared_head_norm ?? output_norm)
/// </code>
/// and the model's LM head turns <c>out</c> into logits. The block's attention
/// and FFN planes are the ones stored on the MTP layer
/// (<c>blk.{n_layer}.*</c>), not shared with the trunk.
/// </summary>
public sealed class MtpHead : IDisposable
{
    private readonly TransformerBlock _block;
    private readonly NormLayer _hnorm;
    private readonly NormLayer _enorm;
    private readonly NormLayer _headNorm;
    private readonly LinearLayer _ehProj;
    private readonly int _hidden;

    public MtpHead(
        TransformerBlock block,
        NormLayer hnorm,
        NormLayer enorm,
        LinearLayer ehProj,
        NormLayer headNorm,
        int hidden)
    {
        ArgumentNullException.ThrowIfNull(block);
        ArgumentNullException.ThrowIfNull(hnorm);
        ArgumentNullException.ThrowIfNull(enorm);
        ArgumentNullException.ThrowIfNull(ehProj);
        ArgumentNullException.ThrowIfNull(headNorm);

        _block = block;
        _hnorm = hnorm;
        _enorm = enorm;
        _ehProj = ehProj;
        _headNorm = headNorm;
        _hidden = hidden;
    }

    /// <summary>
    /// One draft step. <paramref name="trunkHidden"/> is the trunk's final-normed
    /// hidden state for the previous position (<c>t_h_nextn</c>) and
    /// <paramref name="tokenEmbedding"/> is the embedding of the token being
    /// extended; both are <c>[Batch, 1, Hidden]</c>. Returns the head-normed
    /// hidden state <c>[Batch, 1, Hidden]</c>; the caller projects it with the LM
    /// head. <paramref name="position"/> is the absolute position of the token
    /// being extended (its RoPE position and cache slot).
    /// </summary>
    public Tensor<float> Forward(
        Tensor<float> trunkHidden,
        Tensor<float> tokenEmbedding,
        IKVCache? cache,
        int position,
        IWorkspace? workspace = null)
    {
        int batch = trunkHidden.Shape[0];
        if (trunkHidden.Shape[1] != 1 || tokenEmbedding.Shape[1] != 1)
            throw new ArgumentException("MTP drafts one token at a time; expected [Batch, 1, Hidden].");

        using var hNorm = _hnorm.Forward(trunkHidden, workspace);
        using var eNorm = _enorm.Forward(tokenEmbedding, workspace);

        // [e_norm | h_norm] per row, matching ggml_concat(dim 0).
        var concat = workspace != null
            ? workspace.Rent<float>([batch, 1, 2 * _hidden])
            : new Tensor<float>(batch, 1, 2 * _hidden);
        try
        {
            var dst = concat.Data;
            var e = eNorm.Data;
            var h = hNorm.Data;
            for (int b = 0; b < batch; b++)
            {
                e.Slice(b * _hidden, _hidden).CopyTo(dst.Slice(b * 2 * _hidden, _hidden));
                h.Slice(b * _hidden, _hidden).CopyTo(dst.Slice(b * 2 * _hidden + _hidden, _hidden));
            }

            using var projected = _ehProj.Forward(concat, workspace);
            var x = _block.Forward(projected, cache, position, causal: true, workspace, windowSize: 0, perLayerInputs: null);
            return _headNorm.Forward(x, workspace);
        }
        finally
        {
            concat.Dispose();
        }
    }

    public void Dispose()
    {
        _block.Dispose();
        _hnorm.Dispose();
        _enorm.Dispose();
        _headNorm.Dispose();
        _ehProj.Dispose();
    }
}
