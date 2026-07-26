using System.Runtime.CompilerServices;
using JigSawDotNet;
using SharpMind.Core.Embeddings;
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
        public readonly PositionalEncoder PositionalEncoder;
        private NormLayer? _qNorm;
        private NormLayer? _kNorm;
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

    [PuzzleCornerPiece(SharpMindConfig.KeyAttentionQ4,
        SharpMindConfig.ValMhaFlashQ4_0Avx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0AVX2),
        SharpMindConfig.ValMhaFlashQ4_0Fma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0FMA),
        SharpMindConfig.ValMhaFlashQ4_0Scalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0Scalar),
        SharpMindConfig.ValGqaFlashQ4_0Avx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0AVX2),
        SharpMindConfig.ValGqaFlashQ4_0Fma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0FMA),
        SharpMindConfig.ValGqaFlashQ4_0Scalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0Scalar),
        SharpMindConfig.ValMqaFlashQ4_0Avx2, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0AVX2),
        SharpMindConfig.ValMqaFlashQ4_0Fma, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0FMA),
        SharpMindConfig.ValMqaFlashQ4_0Scalar, NS + "." + nameof(AttentionKernels.ScaledDotProductFlashQ4_0Scalar))]
    public abstract unsafe void ScaledDotProductQ4_0(float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope);

    private unsafe void ScaledDotProductForQuantized(QuantDType quantKind,
        float* q, byte* kQuant, byte* vQuant, float* output,
        int seqLen, int kvLen, int headDim, float scale, bool causal,
        int qStride, int oStride, float alibiSlope)
    {
        switch (quantKind)
        {
            case QuantDType.Q4_0:
                ScaledDotProductQ4_0(q, kQuant, vQuant, output, seqLen, kvLen, headDim, scale, causal, qStride, oStride, alibiSlope);
                break;
            case QuantDType.Q8_0:
                ScaledDotProductQ8_0(q, kQuant, vQuant, output, seqLen, kvLen, headDim, scale, causal, qStride, oStride, alibiSlope);
                break;
            default:
                throw new NotSupportedException($"Quantized attention not supported for {quantKind}");
        }
    }

    protected AttentionLayer(ModelConfig config)
        : this(config, null, null)
    {
    }

    protected AttentionLayer(ModelConfig config, TransformerWeights.BlockWeights? weights, Dictionary<string, string>? mapping)
    {
        Config = config;
        int qDim = config.NumHeads * config.HeadDim;
        int kvDim = config.NumKvHeads * config.HeadDim;

        var tm = weights?.TensorMeta;
        if (mapping != null)
        {
            Wq = LinearLayerFactory.Create("q_proj", config.HiddenDim, qDim, true,
                weights?.Wq, weights?.WqBias, tm?.GetValueOrDefault("RawWq").Dtype ?? QuantDType.F32, mapping);
            Wk = LinearLayerFactory.Create("k_proj", config.HiddenDim, kvDim, true,
                weights?.Wk, weights?.WkBias, tm?.GetValueOrDefault("RawWk").Dtype ?? QuantDType.F32, mapping);
            Wv = LinearLayerFactory.Create("v_proj", config.HiddenDim, kvDim, true,
                weights?.Wv, weights?.WvBias, tm?.GetValueOrDefault("RawWv").Dtype ?? QuantDType.F32, mapping);
            Wo = LinearLayerFactory.Create("o_proj", qDim, config.HiddenDim, true,
                weights?.Wo, weights?.WoBias, tm?.GetValueOrDefault("RawWo").Dtype ?? QuantDType.F32, mapping);
        }
        else
        {
            Wq = LinearLayerFactory.Create("q_proj", config.HiddenDim, qDim, true,
                weights?.Wq, weights?.WqBias, tm?.GetValueOrDefault("RawWq").Dtype ?? QuantDType.F32);
            Wk = LinearLayerFactory.Create("k_proj", config.HiddenDim, kvDim, true,
                weights?.Wk, weights?.WkBias, tm?.GetValueOrDefault("RawWk").Dtype ?? QuantDType.F32);
            Wv = LinearLayerFactory.Create("v_proj", config.HiddenDim, kvDim, true,
                weights?.Wv, weights?.WvBias, tm?.GetValueOrDefault("RawWv").Dtype ?? QuantDType.F32);
            Wo = LinearLayerFactory.Create("o_proj", qDim, config.HiddenDim, true,
                weights?.Wo, weights?.WoBias, tm?.GetValueOrDefault("RawWo").Dtype ?? QuantDType.F32);
        }
        PositionalEncoder = config.PositionalEncoding switch
        {
            PositionalEncoding.NoPE => new NoPE(),
            PositionalEncoding.ALiBi => new AlibiEncoder(config.NumHeads),
            _ => new RoPE(config.HeadDim, config.MaxSeqLen, config.RopeTheta,
                 ropeDim: config.RopeDim, ropeScalingType: config.RopeScalingType,
                 ropeScalingFactor: config.RopeScalingFactor),
        };
    }

    public void SetWeights(TransformerWeights.BlockWeights weights)
    {
        // Restore individual Q/K/V layers for fast quantized forward path
        if (weights.Wq != null) Wq.ReplaceWeights(weights.Wq, weights.WqBias);
        Wq.SetRawWeight(weights.RawWq);
        if (weights.Wk != null) Wk.ReplaceWeights(weights.Wk, weights.WkBias);
        Wk.SetRawWeight(weights.RawWk);
        if (weights.Wv != null) Wv.ReplaceWeights(weights.Wv, weights.WvBias);
        Wv.SetRawWeight(weights.RawWv);

        if (weights.Wo != null) Wo.ReplaceWeights(weights.Wo, weights.WoBias);
        Wo.SetRawWeight(weights.RawWo);

        // Per-head Q/K normalization (Qwen3)
        if (weights.QNormW != null)
        {
            _qNorm?.Dispose();
            _qNorm = new RmsNormLayer(Config.HeadDim, Config.NormEps, weights.QNormW, null);
        }
        if (weights.KNormW != null)
        {
            _kNorm?.Dispose();
            _kNorm = new RmsNormLayer(Config.HeadDim, Config.NormEps, weights.KNormW, null);
        }
    }

    public void LoadWeights(string name, ReadOnlySpan<float> data)
    {
        bool isBias = name.EndsWith(".bias", StringComparison.OrdinalIgnoreCase);

        // Q/K norm must be checked BEFORE the broader attn_q/attn_k checks
        if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (_qNorm != null) _qNorm.LoadWeight(data);
            return;
        }
        if (name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (_kNorm != null) _kNorm.LoadWeight(data);
            return;
        }
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

    public bool SetRawWeight(string weightName, byte[] rawData, QuantDType dtype)
    {
        bool isBias = weightName.EndsWith(".bias", StringComparison.OrdinalIgnoreCase);
        if (isBias) return false;

        // Q/K norm is always loaded as float — skip quantized path
        if (weightName.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase) ||
            weightName.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase))
            return false;

        // Force dequantization for Q, K, V
        if (weightName.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || weightName.Contains("q_proj", StringComparison.OrdinalIgnoreCase) ||
            weightName.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || weightName.Contains("k_proj", StringComparison.OrdinalIgnoreCase) ||
            weightName.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || weightName.Contains("v_proj", StringComparison.OrdinalIgnoreCase))
            return false;
            
        if (weightName.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || weightName.Contains("attn_o.", StringComparison.OrdinalIgnoreCase) ||
            weightName.Contains("o_proj", StringComparison.OrdinalIgnoreCase) || weightName.Contains("out_proj", StringComparison.OrdinalIgnoreCase))
        {
            Wo.SetRawWeight(rawData);
            return true;
        }
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

    public Tensor<float> Forward( Tensor<float> x, int positionOffset = 0, bool causal = true, IKVCache? cache = null, Core.Memory.Workspace? workspace = null)
    {
        ThrowIfDisposed();
        int batch = x.Shape[0];
        int seqLen = x.Shape[1];
        int hidden = x.Shape[2];
        int numH = Config.NumHeads;
        int numKv = Config.NumKvHeads;
        int headDim = Config.HeadDim;
        int qDim = numH * headDim;
        int kvDim = numKv * headDim;
        float scale = 1f / MathF.Sqrt(headDim);

        // Individual Q, K, V projections (each uses quantized forward path when available)
        using var q = Wq.Forward(x, workspace);
        using var k = Wk.Forward(x, workspace);
        using var v = Wv.Forward(x, workspace);
        /*if (DumpProjections && _dumpLayerCounter++ < 1)
        {
            void DumpProj(string label, Tensor<float> t)
            {
                double sum = 0; float mn = float.MaxValue, mx = float.MinValue; bool hasN = false;
                for (int i = 0; i < t.ElementCount; i++) { var vv = t.Data[i]; if (float.IsNaN(vv)) hasN = true; if (vv < mn) mn = vv; if (vv > mx) mx = vv; sum += vv; }
                Console.Error.WriteLine($"  {label}: [{t.Shape[0]},{t.Shape[1]}] elems={t.ElementCount} min={mn:G4} max={mx:G4} mean={sum/t.ElementCount:G4} hasNaN={hasN}");
                Console.Error.Write("    first 8: ");
                for (int i = 0; i < Math.Min(8, t.ElementCount); i++) Console.Error.Write($"{t.Data[i]:G4} ");
                Console.Error.WriteLine();
            }
            DumpProj("Q", q); DumpProj("K", k); DumpProj("V", v);
        }*/

        // Apply per-head Q/K normalization (Qwen3):
        // Reshape to [batch*seqLen*nHeads, headDim] so NormLayer normalizes along headDim.
        // Forward always allocates a new tensor; using var disposes it at scope exit.
        // The normed tensor is 2D [totalHeads, headDim]; reshape at use sites below.
        // DEBUG: ForceBypassQKNorm skips normalization to isolate norm-related issues.
        using var qNormed = (_qNorm != null)
            ? _qNorm.Forward(q.Reshape(batch * seqLen * numH, headDim), workspace)
            : null;
        using var kNormed = (_kNorm != null)
            ? _kNorm.Forward(k.Reshape(batch * seqLen * numKv, headDim), workspace)
            : null;

        // Use normed tensors where available, fall back to raw projections
        var qForAttn = (Tensor<float>?)qNormed ?? q;
        var kForAttn = (Tensor<float>?)kNormed ?? k;

        using var qr = qForAttn.Reshape(batch, seqLen, numH, headDim);
        using var kr = kForAttn.Reshape(batch, seqLen, numKv, headDim);
        PositionalEncoder.ApplyBatched(qr, positionOffset);
        PositionalEncoder.ApplyBatched(kr, positionOffset);

        // Use the (possibly normed) K and V for cache storage
        if (cache != null)
        {
            var kCache = kNormed != null ? kNormed.Reshape(batch, seqLen, kvDim) : k;
            cache.Update(kCache, v, numKv, headDim);
        }

        Tensor<float> output = workspace != null
            ? workspace.Rent<float>([batch, seqLen, qDim])
            : new Tensor<float>(batch, seqLen, qDim);
        int effectiveKvLen = cache != null ? cache.Length : seqLen;

        {
            int totalHeads = batch * numH;
            int qStride = numH * headDim;
            int oStride = qDim;

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
                    float* pO = output.DataPtr + (long)(b * seqLen * qDim + h * headDim);

                    if (cache is { IsQuantized: true })
                    {
                        ScaledDotProductForQuantized(cache.QuantKind, pQ,
                            cache.GetQuantizedKeyPtr(b, 0, kvHead),
                            cache.GetQuantizedValuePtr(b, 0, kvHead),
                            pO, seqLen, effectiveKvLen, headDim, scale, causal, qStride, oStride, alibiSlope);
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

        var projected = Wo.Forward(output, workspace);
        output.Dispose();
        return projected;
    }
    /*
    public (Tensor<float> Output, AttentionLayerState State) ForwardWithState(Tensor<float> x, int positionOffset = 0)
    {
        var output = Forward(x, positionOffset);
        var state = new AttentionLayerState { Input = x, Output = output };
        return (output, state);
    }

    public unsafe Tensor<float> Backward(Tensor<float> gradOutput)
    {
        var fn = _qOps.QuantizedMatMulOpFor(QuantDType.F32);

        using var wOutT = Wo.Weight.Transpose();
        var dHidden = new Tensor<float>(gradOutput.Shape.Rows, Wo.InFeatures);
        fn(gradOutput.DataPtr, (byte*)wOutT.DataPtr, dHidden.DataPtr, gradOutput.Shape.Rows, gradOutput.Shape.Cols, Wo.InFeatures);

        using var wQT = Wq.Weight.Transpose();
        var gradInput = new Tensor<float>(dHidden.Shape.Rows, Wq.InFeatures);
        fn(dHidden.DataPtr, (byte*)wQT.DataPtr, gradInput.DataPtr, dHidden.Shape.Rows, dHidden.Shape.Cols, Wq.InFeatures);
        dHidden.Dispose();
        wOutT.Dispose();
        wQT.Dispose();
        return gradInput;
    }
    */

    public IEnumerable<Parameter> Parameters()
    {
        foreach (var p in Wq.Parameters()) yield return p;
        foreach (var p in Wk.Parameters()) yield return p;
        foreach (var p in Wv.Parameters()) yield return p;
        foreach (var p in Wo.Parameters()) yield return p;
        if (_qNorm != null) foreach (var p in _qNorm.Parameters()) yield return p;
        if (_kNorm != null) foreach (var p in _kNorm.Parameters()) yield return p;
    }

    public void FreeFloatWeights()
    {
        Wq.FreeFloatWeight();
        Wk.FreeFloatWeight();
        Wv.FreeFloatWeight();
        Wo.FreeFloatWeight();
    }

    public void Dispose() { Dispose(true); GC.SuppressFinalize(this); }
    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            // These are LinearLayers, they will only dispose if they own the weights.
            Wq.Dispose(); Wk.Dispose(); Wv.Dispose(); Wo.Dispose();
            _qNorm?.Dispose(); _kNorm?.Dispose();
        }
        _disposed = true;
    }
    ~AttentionLayer() => Dispose(false);
    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, nameof(AttentionLayer));
}
