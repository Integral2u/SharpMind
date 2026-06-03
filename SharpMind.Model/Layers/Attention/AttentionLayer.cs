using JigSawDotNet;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;

namespace SharpMind.Model.Layers.Attention;

public abstract class AttentionLayer : IDisposable
{
    private const string NS = $"{nameof(SharpMind)}.{nameof(Model)}.{nameof(Layers)}.{nameof(Attention)}.{nameof(AttentionKernels)}";

    protected readonly ModelConfig Config;
    public readonly LinearLayer Wq;
    public readonly LinearLayer Wk;
    public readonly LinearLayer Wv;
    public readonly LinearLayer Wo;
    public readonly RoPE Rope;
    private bool _disposed;

    protected AttentionLayer(ModelConfig config, QuantizationOps qOps)
    {
        Config = config;
        int kvDim = config.NumKvHeads * config.HeadDim;
        Wq = new LinearLayer("q_proj", config.HiddenDim, config.HiddenDim, bias: true, qOps: qOps);
        Wk = new LinearLayer("k_proj", config.HiddenDim, kvDim, bias: true, qOps: qOps);
        Wv = new LinearLayer("v_proj", config.HiddenDim, kvDim, bias: true, qOps: qOps);
        Wo = new LinearLayer("o_proj", config.HiddenDim, config.HiddenDim, bias: true, qOps: qOps);
        Rope = new RoPE(config.HeadDim, config.MaxSeqLen, config.RopeTheta);
    }

    public void LoadWeights(string name, ReadOnlySpan<float> data)
    {
        var lower = name.ToLower();
        bool isBias = lower.EndsWith(".bias");

        // Use full tensor-name suffixes — single-char matching ("q", "k", "v") is
        // fooled by the "blk" block prefix which contains "k", causing V weights to
        // silently load into Wk and leaving Wv permanently zeroed.
        if (lower.Contains("attn_q") || lower.Contains("q_proj"))
        {
            if (isBias) Wq.LoadBias(data);
            else Wq.LoadWeightTransposed(data);
        }
        else if (lower.Contains("attn_k") || lower.Contains("k_proj"))
        {
            if (isBias) Wk.LoadBias(data);
            else Wk.LoadWeightTransposed(data);
        }
        else if (lower.Contains("attn_v") || lower.Contains("v_proj"))
        {
            if (isBias) Wv.LoadBias(data);
            else Wv.LoadWeightTransposed(data);
        }
        else if (lower.Contains("attn_output") || lower.Contains("attn_o.") ||
                 lower.Contains("o_proj") || lower.Contains("out_proj"))
        {
            if (isBias) Wo.LoadBias(data);
            else Wo.LoadWeightTransposed(data);
        }
    }

    public bool SetRawWeight(string weightName, byte[] rawData, Format.GgufDtype dtype)
    {
        var lower = weightName.ToLower();
        bool isBias = lower.EndsWith(".bias");
        if (isBias) return false;

        if (lower.Contains("attn_q") || lower.Contains("q_proj"))
            return Wq.SetRawWeight(rawData, dtype);
        if (lower.Contains("attn_k") || lower.Contains("k_proj"))
            return Wk.SetRawWeight(rawData, dtype);
        if (lower.Contains("attn_v") || lower.Contains("v_proj"))
            return Wv.SetRawWeight(rawData, dtype);
        if (lower.Contains("attn_output") || lower.Contains("attn_o.") ||
            lower.Contains("o_proj") || lower.Contains("out_proj"))
            return Wo.SetRawWeight(rawData, dtype);
        return false;
    }

    [PuzzleCornerPiece(SharpMindConfig.KeyAttention,
        SharpMindConfig.ValMhaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValMhaFma, NS + "." + nameof(AttentionKernels.ScaledDotProductFMA),
        SharpMindConfig.ValMhaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValMhaFlashAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashAVX2),
        SharpMindConfig.ValMhaFlashFma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashFMA),
        SharpMindConfig.ValMhaFlashScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashScalar),
        SharpMindConfig.ValGqaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValGqaFma, NS + "." + nameof(AttentionKernels.ScaledDotProductFMA),
        SharpMindConfig.ValGqaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValGqaFlashAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashAVX2),
        SharpMindConfig.ValGqaFlashFma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashFMA),
        SharpMindConfig.ValGqaFlashScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashScalar),
        SharpMindConfig.ValMqaAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductAVX2),
        SharpMindConfig.ValMqaFma, NS + "." + nameof(AttentionKernels.ScaledDotProductFMA),
        SharpMindConfig.ValMqaScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductScalar),
        SharpMindConfig.ValMqaFlashAvx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashAVX2),
        SharpMindConfig.ValMqaFlashFma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashFMA),
        SharpMindConfig.ValMqaFlashScalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashScalar))]
    public abstract unsafe void ScaledDotProduct(float* q, float* k, float* v, float* o, int seqLen, int kvLen, int headDim, float scale, bool causal, int qStride, int oStride);

    public Tensor<float> Forward(
        Tensor<float> x,
        TensorOps ops,
        int positionOffset = 0,
        bool causal = true,
        IKVCache? cache = null)
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

        cache?.Update(k, v, numKv, headDim);

        var output = new Tensor<float>(batch, seqLen, hidden);
        int effectiveKvLen = cache != null ? cache.Length : seqLen;

        {
            int totalHeads = batch * numH;
            int qStride = numH * headDim;
            int oStride = hidden;

            void DoHead(int bh)
            {
                int b = bh / numH;
                int h = bh % numH;
                int kvHead = h / Config.KvGroupSize;
                unsafe
                {
                    float* pQ = qr.DataPtr + (long)(b * seqLen * numH + h) * headDim;
                    float* pO = output.DataPtr + (long)(b * seqLen * hidden + h * headDim);

                    float* pK;
                    float* pV;
                    if (cache is KVCache kc)
                    {
                        int cacheStride = kc.AllocatedCapacity;
                        pK = kc.Keys.DataPtr + (long)b * (numKv * cacheStride * headDim)
                                           + (long)kvHead * (cacheStride * headDim);
                        pV = kc.Values.DataPtr + (long)b * (numKv * cacheStride * headDim)
                                              + (long)kvHead * (cacheStride * headDim);
                    }
                    else if (cache != null)
                    {
                        throw new NotSupportedException(
                            $"Non-contiguous cache type {cache.GetType().Name} is not supported. " +
                            "Use KVCache for contiguous pointer access.");
                    }
                    else
                    {
                        using var kHead = new Tensor<float>(effectiveKvLen, headDim);
                        using var vHead = new Tensor<float>(effectiveKvLen, headDim);
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

                    ScaledDotProduct(pQ, pK, pV, pO, seqLen, effectiveKvLen, headDim, scale, causal, qStride, oStride);
                }
            }

            if (totalHeads <= 16)
            {
                for (int bh = 0; bh < totalHeads; bh++) DoHead(bh);
            }
            else
            {
                Parallel.For(0, totalHeads, DoHead);
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

    public Tensor<float> Backward(Tensor<float> gradOutput, TensorOps ops)
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
