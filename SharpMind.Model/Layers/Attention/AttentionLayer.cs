using JigSawDotNet;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Ops;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Attention;

public abstract class AttentionLayer : IDisposable
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(Attention)}.{nameof(AttentionKernels)}";

    protected readonly ModelConfig Config;
    protected readonly LinearLayer Wq;
    protected readonly LinearLayer Wk;
    protected readonly LinearLayer Wv;
    protected readonly LinearLayer Wo;
    protected readonly RoPE Rope;
    private bool _disposed;

    protected AttentionLayer(ModelConfig config)
    {
        Config = config;
        int kvDim = config.NumKvHeads * config.HeadDim;
        Wq = new LinearLayer("q_proj", config.HiddenDim, config.HiddenDim, bias: true);
        Wk = new LinearLayer("k_proj", config.HiddenDim, kvDim, bias: true);
        Wv = new LinearLayer("v_proj", config.HiddenDim, kvDim, bias: true);
        Wo = new LinearLayer("o_proj", config.HiddenDim, config.HiddenDim, bias: true);
        Rope = new RoPE(config.HeadDim, config.MaxSeqLen, config.RopeTheta);
    }

    [PuzzleCornerPiece(SharpMindConfig.KeyAttention,
        SharpMindConfig.ValMhaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValMhaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValGqaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValGqaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValMqaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValMqaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar))]
    public abstract unsafe void ScaledDotProduct(float* q, float* k, float* v, float* o, int seqLen, int kvLen, int headDim, float scale, bool causal);

    public Tensor<float> Forward(
        Tensor<float> x,
        TensorOps ops,
        int positionOffset = 0,
        bool causal = true,
        KVCache? cache = null)
    {
        ThrowIfDisposed();
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hidden = x.Shape[2];
        int numH = Config.NumHeads;
        int numKv = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int kvDim = numKv * headDim;
        float scale = 1f / MathF.Sqrt(headDim);

        using var q = Wq.Forward(x, ops);
        using var k = Wk.Forward(x, ops);
        using var v = Wv.Forward(x, ops);

        using var qr = q.Reshape(batch, seqLen, numH, headDim);
        using var kr = k.Reshape(batch, seqLen, numKv, headDim);
        Rope.ApplyBatched(qr, positionOffset);
        Rope.ApplyBatched(kr, positionOffset);

        if (cache != null)
        {
            cache.Update(k, v, numKv, headDim);
        }

        var output = new Tensor<float>(batch, seqLen, hidden);
        int effectiveKvLen = cache != null ? cache.CurrentPosition : seqLen;

        unsafe
        {
            for (int b = 0; b < batch; b++)
            {
                for (int h = 0; h < numH; h++)
                {
                    int kvHead = h / Config.KvGroupSize;
                    using var qHead = new Tensor<float>(seqLen, headDim);
                    using var oHead = new Tensor<float>(seqLen, headDim);
                    Tensor<float>? kHead = null;
                    Tensor<float>? vHead = null;
                    try
                    {
                        // Pack Q head into contiguous [SeqLen, HeadDim] buffer.
                        for (int s = 0; s < seqLen; s++)
                        {
                            float* srcQ = qr.DataPtr + (long)((b * seqLen + s) * numH + h) * headDim;
                            float* dstQ = qHead.DataPtr + (long)s * headDim;
                            for (int d = 0; d < headDim; d++) dstQ[d] = srcQ[d];
                        }

                        float* pK;
                        float* pV;
                        if (cache != null)
                        {
                            pK = cache.Keys.DataPtr + (long)b * (numKv * Config.MaxSeqLen * headDim)
                                               + (long)kvHead * (Config.MaxSeqLen * headDim);
                            pV = cache.Values.DataPtr + (long)b * (numKv * Config.MaxSeqLen * headDim)
                                                 + (long)kvHead * (Config.MaxSeqLen * headDim);
                        }
                        else
                        {
                            // Pack K/V heads into contiguous [SeqLen, HeadDim] buffers.
                            kHead = new Tensor<float>(effectiveKvLen, headDim);
                            vHead = new Tensor<float>(effectiveKvLen, headDim);
                            for (int s = 0; s < effectiveKvLen; s++)
                            {
                                float* srcK = kr.DataPtr + (long)((b * seqLen + s) * numKv + kvHead) * headDim;
                                float* srcV = v.DataPtr + (long)(b * seqLen * kvDim + s * kvDim + kvHead * headDim);
                                float* dstK = kHead.DataPtr + (long)s * headDim;
                                float* dstV = vHead.DataPtr + (long)s * headDim;
                                for (int d = 0; d < headDim; d++)
                                {
                                    dstK[d] = srcK[d];
                                    dstV[d] = srcV[d];
                                }
                            }
                            pK = kHead.DataPtr;
                            pV = vHead.DataPtr;
                        }

                        ScaledDotProduct(qHead.DataPtr, pK, pV, oHead.DataPtr, seqLen, effectiveKvLen, headDim, scale, causal);

                        // Scatter contiguous head output back into [B, SeqLen, Hidden].
                        for (int s = 0; s < seqLen; s++)
                        {
                            float* srcO = oHead.DataPtr + (long)s * headDim;
                            float* dstO = output.DataPtr + (long)(b * seqLen * hidden + s * hidden + h * headDim);
                            for (int d = 0; d < headDim; d++) dstO[d] = srcO[d];
                        }
                    }
                    finally
                    {
                        kHead?.Dispose();
                        vHead?.Dispose();
                    }
                }
            }
        }

        var projected = Wo.Forward(output, ops);
        output.Dispose();
        return projected;
    }

    public (Tensor<float> Output, AttentionLayerState State) ForwardWithState(Tensor<float> x, TensorOps ops, int positionOffset = 0)
    {
        var output = Forward(x, ops, positionOffset);
        var state = new AttentionLayerState { Input = x, Output = output };
        return (output, state);
    }

    public Tensor<float> Backward(Tensor<float> gradOutput, AttentionLayerState state, TensorOps ops)
    {
        // Simplified: propagate gradient through output projection
        // Full backward requires full attention kernel gradients - stub for now
        using var wOutT = TensorOps.Transpose(Wo.Weight);
        var dHidden = ops.MatMul(gradOutput, wOutT);
        using var wQT = TensorOps.Transpose(Wq.Weight);
        var gradInput = ops.MatMul(dHidden, wQT);
        dHidden.Dispose();
        wOutT.Dispose();
        wQT.Dispose();
        return gradInput;
    }

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var p in Wq.Parameters()) yield return p;
        foreach (var p in Wk.Parameters()) yield return p;
        foreach (var p in Wv.Parameters()) yield return p;
        foreach (var p in Wo.Parameters()) yield return p;
    }

    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing) { Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose(); }
        _disposed = true;
    }
    ~AttentionLayer() => Dispose(false);
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(AttentionLayer));
}

public class AttentionLayerState
{
    public Tensor<float> Input { get; init; } = null!;
    public Tensor<float> Output { get; init; } = null!;
}