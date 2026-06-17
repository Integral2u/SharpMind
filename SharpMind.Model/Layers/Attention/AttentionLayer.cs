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
    public readonly LinearLayer Wqkv; // Added fused layer
    public readonly PositionalEncoder PositionalEncoder;
    private bool _disposed;

    [PuzzleCornerPiece(SharpMindConfig.KeyAttentionQ8,
        SharpMindConfig.ValMhaFlashQ8_0Avx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0AVX2),
        SharpMindConfig.ValMhaFlashQ8_0Fma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0FMA),
        SharpMindConfig.ValMhaFlashQ8_0Scalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0Scalar),
        SharpMindConfig.ValGqaFlashQ8_0Avx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0AVX2),
        SharpMindConfig.ValGqaFlashQ8_0Fma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0FMA),
        SharpMindConfig.ValGqaFlashQ8_0Scalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0Scalar),
        SharpMindConfig.ValMqaFlashQ8_0Avx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0AVX2),
        SharpMindConfig.ValMqaFlashQ8_0Fma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0FMA),
        SharpMindConfig.ValMqaFlashQ8_0Scalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ8_0Scalar))]
    public abstract unsafe void ScaledDotProductQ8_0(float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope);

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
        
        // Initialize Wqkv layer
        int totalQkvDim = config.HiddenDim + 2 * kvDim;
        Wqkv = new LinearLayer("qkv_proj", config.HiddenDim, totalQkvDim, bias: true, qOps: qOps, null, null);
        PositionalEncoder = config.PositionalEncoding switch
        {
            PositionalEncoding.NoPE => new NoPE(),
            PositionalEncoding.ALiBi => new AlibiEncoder(config.NumHeads),
            _ => new RoPE(config.HeadDim, config.MaxSeqLen, config.RopeTheta),
        };

        if (weights != null)
            CopyFusedWeights(weights.Wq, weights.Wk, weights.Wv, weights.WqBias, weights.WkBias, weights.WvBias);
    }

    private void CopyFusedWeights(Tensor<float> wq, Tensor<float> wk, Tensor<float> wv,
        Tensor<float>? wqB, Tensor<float>? wkB, Tensor<float>? wvB)
    {
        int hiddenDim = Config.HiddenDim;
        int kvDim = Config.NumKvHeads * Config.HeadDim;
        int totalOut = hiddenDim + 2 * kvDim;
        var wData = Wqkv.Weight.Data;

        // Weights are stored transposed [Out, In] in LinearLayer
        for (int i = 0; i < hiddenDim; i++) // OutFeatures
        {
            for (int j = 0; j < hiddenDim; j++) // InFeatures
                wData[i * totalOut + j] = wq.Data[j * hiddenDim + i];
        }
        for (int i = 0; i < kvDim; i++) // OutFeatures
        {
            for (int j = 0; j < hiddenDim; j++) // InFeatures
            {
                wData[(hiddenDim + i) * totalOut + j] = wk.Data[j * kvDim + i];
                wData[(hiddenDim + kvDim + i) * totalOut + j] = wv.Data[j * kvDim + i];
            }
        }

        if (Wqkv.Bias != null)
        {
            wqB?.Data.CopyTo(Wqkv.Bias.Data[..hiddenDim]);
            wkB?.Data.CopyTo(Wqkv.Bias.Data.Slice(hiddenDim, kvDim));
            wvB?.Data.CopyTo(Wqkv.Bias.Data.Slice(hiddenDim + kvDim, kvDim));
        }
    }

    public void SetWeights(TransformerWeights.BlockWeights weights)
    {
        int kvDim = Config.NumKvHeads * Config.HeadDim;
        int hiddenDim = Config.HiddenDim;
        int totalOut = hiddenDim + 2 * kvDim;
        var wData = Wqkv.Weight.Data;

        // Weights are stored transposed [Out, In]
        // Fused Wqkv is [In, Out]
        for (int i = 0; i < hiddenDim; i++) // InFeatures
        {
            for (int j = 0; j < hiddenDim; j++) // OutFeatures
                wData[j * totalOut + i] = weights.Wq.Data[i * hiddenDim + j];
        }
        for (int i = 0; i < kvDim; i++) // OutFeatures
        {
            for (int j = 0; j < hiddenDim; j++) // InFeatures
            {
                wData[j * totalOut + (hiddenDim + i)] = weights.Wk.Data[i * kvDim + j];
                wData[j * totalOut + (hiddenDim + kvDim + i)] = weights.Wv.Data[i * kvDim + j];
            }
        }

        if (Wqkv.Bias != null && weights.WqBias != null)
        {
            weights.WqBias.Data.CopyTo(Wqkv.Bias.Data[..hiddenDim]);
            weights.WkBias.Data.CopyTo(Wqkv.Bias.Data.Slice(hiddenDim, kvDim));
            weights.WvBias.Data.CopyTo(Wqkv.Bias.Data.Slice(hiddenDim + kvDim, kvDim));
        }

        // Restore individual Q/K/V layers for fast quantized forward path
        Wq.ReplaceWeights(weights.Wq, weights.WqBias);
        Wq.SetRawWeight(weights.RawWq, weights.QuantDtype ?? GgufDtype.F32);
        Wk.ReplaceWeights(weights.Wk, weights.WkBias);
        Wk.SetRawWeight(weights.RawWk, weights.QuantDtype ?? GgufDtype.F32);
        Wv.ReplaceWeights(weights.Wv, weights.WvBias);
        Wv.SetRawWeight(weights.RawWv, weights.QuantDtype ?? GgufDtype.F32);

        Wo.ReplaceWeights(weights.Wo, weights.WoBias);
        Wo.SetRawWeight(weights.RawWo, weights.QuantDtype ?? GgufDtype.F32);
    }

    private unsafe void LoadFusedWeightTransposed(ReadOnlySpan<float> data, int colOffset, int subOutF)
    {
        // data is GGUF layout [subOutF, HiddenDim] (transposed)
        // Wqkv.Weight is [HiddenDim, totalOut] (Row-major: [In, Out])
        var w = Wqkv.Weight;
        int inF = w.Shape[0]; // HiddenDim
        int totalOut = w.Shape[1]; // TotalOut
        var wData = w.Data;

        for (int o = 0; o < subOutF; o++) // OutFeatures index in sub-tensor
        {
            for (int i = 0; i < inF; i++) // InFeatures
            {
                // wData[row * totalOut + col]
                // row = i, col = colOffset + o
                wData[i * totalOut + (colOffset + o)] = data[o * inF + i];
            }
        }
        // Stale _weightBT will be recomputed on next Forward
    }

    public void LoadWeights(string name, ReadOnlySpan<float> data)
    {
        bool isBias = name.EndsWith(".bias", StringComparison.OrdinalIgnoreCase);
        int hiddenDim = Config.HiddenDim;
        int kvDim = Config.NumKvHeads * Config.HeadDim;

        if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias) data.CopyTo(Wqkv.Bias!.Data[..hiddenDim]);
            else LoadFusedWeightTransposed(data, 0, hiddenDim);
        }
        else if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias) data.CopyTo(Wqkv.Bias!.Data.Slice(hiddenDim, kvDim));
            else LoadFusedWeightTransposed(data, hiddenDim, kvDim);
        }
        else if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase))
        {
            if (isBias) data.CopyTo(Wqkv.Bias!.Data.Slice(hiddenDim + kvDim, kvDim));
            else LoadFusedWeightTransposed(data, hiddenDim + kvDim, kvDim);
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

        // Force dequantization for Q, K, V
        if (weightName.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || weightName.Contains("q_proj", StringComparison.OrdinalIgnoreCase) ||
            weightName.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || weightName.Contains("k_proj", StringComparison.OrdinalIgnoreCase) ||
            weightName.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || weightName.Contains("v_proj", StringComparison.OrdinalIgnoreCase))
            return false;
            
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
    public abstract unsafe void ScaledDotProduct(float* q, float* k, float* v, float* o, int seqLen, int kvLen, int headDim, float scale, bool causal, int qStride, int oStride, float alibiSlope);

    public Tensor<float> Forward( Tensor<float> x, TensorOps ops, int positionOffset = 0, bool causal = true, IKVCache? cache = null, Core.Memory.Workspace? workspace = null)
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

        // Individual Q, K, V projections (each uses quantized forward path when available)
        using var q = Wq.Forward(x, ops, workspace);
        using var k = Wk.Forward(x, ops, workspace);
        using var v = Wv.Forward(x, ops, workspace);

        using var qr = q.Reshape(batch, seqLen, numH, headDim);
        using var kr = k.Reshape(batch, seqLen, numKv, headDim);
        PositionalEncoder.ApplyBatched(qr, positionOffset);
        PositionalEncoder.ApplyBatched(kr, positionOffset);

        cache?.Update(k, v, numKv, headDim);

        Tensor<float> output = workspace != null
            ? workspace.Rent<float>([batch, seqLen, hidden])
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
                    ? workspace.Rent<float>([totalHeads, effectiveKvLen, headDim])
                    : new Tensor<float>(totalHeads, effectiveKvLen, headDim);
                allTempV = workspace != null
                    ? workspace.Rent<float>([totalHeads, effectiveKvLen, headDim])
                    : new Tensor<float>(totalHeads, effectiveKvLen, headDim);
            }

            void DoHead(int bh)
            {
                int b = bh / numH;
                int h = bh % numH;
                int kvHead = h / Config.KvGroupSize;
                float alibiSlope = PositionalEncoder is AlibiEncoder alibi ? alibi.Slopes[h] : 0f;
                unsafe
                {
                    float* pQ = qr.DataPtr + (long)(b * seqLen * numH + h) * headDim;
                    float* pO = output.DataPtr + (long)(b * seqLen * hidden + h * headDim);

                    if (cache is { IsQuantized: true })
                    {
                        byte* pKQ = cache.GetQuantizedKeyPtr(b, 0, kvHead);
                        byte* pVQ = cache.GetQuantizedValuePtr(b, 0, kvHead);
                        ScaledDotProductQ8_0(pQ, pKQ, pVQ, pO, seqLen, effectiveKvLen, headDim, scale, causal, qStride, oStride, alibiSlope);
                    }
                    else if (cache is { IsContiguous: true })
                    {
                        float* pK = cache.GetKeyPtr(b, 0, kvHead);
                        float* pV = cache.GetValuePtr(b, 0, kvHead);
                        ScaledDotProduct(pQ, pK, pV, pO, seqLen, effectiveKvLen, headDim, scale, causal, qStride, oStride, alibiSlope);
                    }
                    else
                    {
                        float* pK = allTempK!.DataPtr + (long)bh * effectiveKvLen * headDim;
                        float* pV = allTempV!.DataPtr + (long)bh * effectiveKvLen * headDim;

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

                        ScaledDotProduct(pQ, pK, pV, pO, seqLen, effectiveKvLen, headDim, scale, causal, qStride, oStride, alibiSlope);
                    }
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
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose(); Wqkv.Dispose();
        }
        _disposed = true;
    }
    ~AttentionLayer() => Dispose(false);
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(AttentionLayer));
}
