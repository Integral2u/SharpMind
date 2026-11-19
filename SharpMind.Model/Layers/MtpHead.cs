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
    /// hidden state for the previous position(s) (<c>t_h_nextn</c>) and
    /// <paramref name="tokenEmbedding"/> is the embedding of the token(s) being
    /// extended; both are <c>[Batch, SeqLen, Hidden]</c>. A one-token draft uses
    /// <c>SeqLen = 1</c>, while prefill feeds a whole prompt chunk. Returns the
    /// head-normed hidden state <c>[Batch, SeqLen, Hidden]</c>; the caller
    /// projects it with the LM head. <paramref name="position"/> is the absolute
    /// position of the first token (its RoPE position and cache slot).
    /// </summary>
    public Tensor<float> Forward(
        Tensor<float> trunkHidden,
        Tensor<float> tokenEmbedding,
        IKVCache? cache,
        int position,
        IWorkspace? workspace = null)
    {
        var x = RunBlock(trunkHidden, tokenEmbedding, cache, position, workspace);
        try
        {
            return _headNorm.Forward(x, workspace);
        }
        finally
        {
            x.Dispose();
        }
    }

    /// <summary>
    /// Runs the draft block over a chunk without projecting or normalising its
    /// output. Used to fill the head's cache over prompt tokens: only the block's
    /// keys/values matter, and skipping the output norm and LM head keeps prompt
    /// prefill cheap.
    /// </summary>
    public void ForwardFill(
        Tensor<float> trunkHidden,
        Tensor<float> tokenEmbedding,
        IKVCache? cache,
        int position,
        IWorkspace? workspace = null)
    {
        RunBlock(trunkHidden, tokenEmbedding, cache, position, workspace).Dispose();
    }

    private Tensor<float> RunBlock(
        Tensor<float> trunkHidden,
        Tensor<float> tokenEmbedding,
        IKVCache? cache,
        int position,
        IWorkspace? workspace)
    {
        int batch = trunkHidden.Shape[0];
        int seqLen = trunkHidden.Shape[1];
        if (tokenEmbedding.Shape[0] != batch || tokenEmbedding.Shape[1] != seqLen)
            throw new ArgumentException(
                "MTP expects trunkHidden and tokenEmbedding to share [Batch, SeqLen].");
        if (trunkHidden.Shape[2] != _hidden || tokenEmbedding.Shape[2] != _hidden)
            throw new ArgumentException($"MTP expects a hidden width of {_hidden}.");

        using var hNorm = _hnorm.Forward(trunkHidden, workspace);
        using var eNorm = _enorm.Forward(tokenEmbedding, workspace);

        // [e_norm | h_norm] per token, matching ggml_concat(dim 0).
        var concat = workspace != null
            ? workspace.Rent<float>([batch, seqLen, 2 * _hidden])
            : new Tensor<float>(batch, seqLen, 2 * _hidden);
        try
        {
            var dst = concat.Data;
            var e = eNorm.Data;
            var h = hNorm.Data;
            for (int b = 0; b < batch; b++)
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int src = (b * seqLen + s) * _hidden;
                    int dstOffset = (b * seqLen + s) * 2 * _hidden;
                    e.Slice(src, _hidden).CopyTo(dst.Slice(dstOffset, _hidden));
                    h.Slice(src, _hidden).CopyTo(dst.Slice(dstOffset + _hidden, _hidden));
                }
            }

            var projected = _ehProj.Forward(concat, workspace);
            try
            {
                return _block.Forward(projected, cache, position, causal: true, workspace, windowSize: 0, perLayerInputs: null);
            }
            catch
            {
                projected.Dispose();
                throw;
            }
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
