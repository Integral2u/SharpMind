using System.Runtime.CompilerServices;
using JigSawDotNet;
using SharpMind.Core.Embeddings;
using SharpMind.Core.Ops;
using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Core.Training;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

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
        : this(config, qOps, null)
    {
    }

    protected AttentionLayer(ModelConfig config, QuantizationOps qOps, TransformerWeights.BlockWeights? weights)
    {
        Config = config;
        int kvDim = config.NumKvHeads * config.HeadDim;
        Wq = new LinearLayer("q_proj", config.HiddenDim, config.HiddenDim, bias: true, qOps: qOps, weights?.Wq, weights?.WqBias);
        Wk = new LinearLayer("k_proj", config.HiddenDim, kvDim, bias: true, qOps: qOps, weights?.Wk, weights?.WkBias);
        Wv = new LinearLayer("v_proj", config.HiddenDim, kvDim, bias: true, qOps: qOps, weights?.Wv, weights?.WvBias);
        Wo = new LinearLayer("o_proj", config.HiddenDim, config.HiddenDim, bias: true, qOps: qOps, weights?.Wo, weights?.WoBias);
        Rope = new RoPE(config.HeadDim, config.MaxSeqLen, config.RopeTheta);
    }

    public void SetWeights(TransformerWeights.BlockWeights weights)
    {
        Wq.ReplaceWeights(weights.Wq, weights.WqBias);
        Wq.SetRawWeight(weights.RawWq, weights.QuantDtype ?? GgufDtype.F32);
        Wk.ReplaceWeights(weights.Wk, weights.WkBias);
        Wk.SetRawWeight(weights.RawWk, weights.QuantDtype ?? GgufDtype.F32);
        Wv.ReplaceWeights(weights.Wv, weights.WvBias);
        Wv.SetRawWeight(weights.RawWv, weights.QuantDtype ?? GgufDtype.F32);
        Wo.ReplaceWeights(weights.Wo, weights.WoBias);
        Wo.SetRawWeight(weights.RawWo, weights.QuantDtype ?? GgufDtype.F32);
    }

    public void LoadWeights(string name, ReadOnlySpan<float> data)
    {
        bool isBias = name.EndsWith(".bias", StringComparison.OrdinalIgnoreCase);

        // Use full tensor-name suffixes — single-char matching ("q", "k", "v") is
        // fooled by the "blk" block prefix which contains "k", causing V weights to
        // silently load into Wk and leaving Wv permanently zeroed.
        if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias) Wq.LoadBias(data);
            else Wq.LoadWeightTransposed(data);
        }
        else if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias) Wk.LoadBias(data);
            else Wk.LoadWeightTransposed(data);
        }
        else if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias) Wv.LoadBias(data);
            else Wv.LoadWeightTransposed(data);
        }
        else if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("attn_o.", StringComparison.OrdinalIgnoreCase) ||
                 name.Contains("o_proj", StringComparison.OrdinalIgnoreCase) || name.Contains("out_proj", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias) Wo.LoadBias(data);
            else Wo.LoadWeightTransposed(data);
        }
    }

    public bool SetRawWeight(string weightName, byte[] rawData, Format.GgufDtype dtype)
    {
        bool isBias = weightName.EndsWith(".bias", StringComparison.OrdinalIgnoreCase);
        if (isBias) return false;

        if (weightName.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || weightName.Contains("q_proj", StringComparison.OrdinalIgnoreCase))
            return Wq.SetRawWeight(rawData, dtype);
        if (weightName.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || weightName.Contains("k_proj", StringComparison.OrdinalIgnoreCase))
            return Wk.SetRawWeight(rawData, dtype);
        if (weightName.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || weightName.Contains("v_proj", StringComparison.OrdinalIgnoreCase))
            return Wv.SetRawWeight(rawData, dtype);
        if (weightName.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || weightName.Contains("attn_o.", StringComparison.OrdinalIgnoreCase) ||
            weightName.Contains("o_proj", StringComparison.OrdinalIgnoreCase) || weightName.Contains("out_proj", StringComparison.OrdinalIgnoreCase))
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
        IKVCache? cache = null,
        SharpMind.Core.Memory.Workspace? workspace = null)
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

        using var q = Wq.Forward(x, ops, workspace);
        using var k = Wk.Forward(x, ops, workspace);
        using var v = Wv.Forward(x, ops, workspace);

        using var qr = q.Reshape(batch, seqLen, numH, headDim);
        using var kr = k.Reshape(batch, seqLen, numKv, headDim);
        Rope.ApplyBatched(qr, positionOffset);
        Rope.ApplyBatched(kr, positionOffset);

        cache?.Update(k, v, numKv, headDim);

        Tensor<float> output = workspace != null 
            ? workspace.Rent<float>(new[] { batch, seqLen, hidden }) 
            : new Tensor<float>(batch, seqLen, hidden);
        int effectiveKvLen = cache != null ? cache.Length : seqLen;

        {
            int totalHeads = batch * numH;
            int qStride = numH * headDim;
            int oStride = hidden;

            // Pre-allocate temporary buffers for non-contiguous caches to avoid race conditions in Parallel.For
            Tensor<float>? allTempK = null;
            Tensor<float>? allTempV = null;
            if (cache is not { IsContiguous: true })
            {
                allTempK = workspace != null 
                    ? workspace.Rent<float>(new[] { totalHeads, effectiveKvLen, headDim }) 
                    : new Tensor<float>(totalHeads, effectiveKvLen, headDim);
                allTempV = workspace != null 
                    ? workspace.Rent<float>(new[] { totalHeads, effectiveKvLen, headDim }) 
                    : new Tensor<float>(totalHeads, effectiveKvLen, headDim);
            }

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

                    if (cache is { IsContiguous: true })
                    {
                        pK = cache.GetKeyPtr(b, 0, kvHead);
                        pV = cache.GetValuePtr(b, 0, kvHead);
                    }
                    else
                    {
                        pK = allTempK!.DataPtr + (long)bh * effectiveKvLen * headDim;
                        pV = allTempV!.DataPtr + (long)bh * effectiveKvLen * headDim;

                        if (cache != null)
                        {
                            for (int s = 0; s < effectiveKvLen; s++)
                            {
                                float* srcK = cache.GetKeyPtr(b, s, kvHead);
                                float* srcV = cache.GetValuePtr(b, s, kvHead);
                                Unsafe.CopyBlock(pK + (long)s * headDim, srcK, (uint)(headDim * sizeof(float)));
                                Unsafe.CopyBlock(pV + (long)s * headDim, srcV, (uint)(headDim * sizeof(float)));
                            }
                        }
                        else
                        {
                            for (int s = 0; s < effectiveKvLen; s++)
                            {
                                float* srcK = kr.DataPtr + (long)((b * seqLen + s) * numKv + kvHead) * headDim;
                                float* srcV = v.DataPtr + (long)(b * seqLen * kvDim + s * kvDim + kvHead * headDim);
                                Unsafe.CopyBlock(pK + (long)s * headDim, srcK, (uint)(headDim * sizeof(float)));
                                Unsafe.CopyBlock(pV + (long)s * headDim, srcV, (uint)(headDim * sizeof(float)));
                            }
                        }
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

            allTempK?.Dispose();
            allTempV?.Dispose();
        }

        var projected = Wo.Forward(output, ops, workspace);
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
        if (disposing) 
        { 
            // These are LinearLayers, they will only dispose if they own the weights.
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose(); 
        }
        _disposed = true;
    }
    ~AttentionLayer() => Dispose(false);
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(AttentionLayer));
}
