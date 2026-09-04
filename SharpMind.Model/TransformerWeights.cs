using SharpMind.Core.Quantization;
using SharpMind.Core.Tensors;
using SharpMind.Model.Config;
using SharpMind.Model.Format;

namespace SharpMind.Model;

/// <summary>Metadata for one weight tensor: file offset, byte size, quantization dtype.</summary>
public readonly record struct TensorMeta(long Offset, int Size, QuantDType Dtype);

/// <summary>Container for a Transformer's weights — all weights loaded into memory.</summary>
public abstract class TransformerWeights : IDisposable
{
    public ModelConfig Config { get; }
    public Tensor<float> EmbeddingWeight { get; }
    public Tensor<float>? LmHeadWeight { get; protected set; }
    public void SetLmHead(Tensor<float> head) => LmHeadWeight = head;
    public Tensor<float> FinalNormWeight { get; }
    public Tensor<float>? FinalNormBias { get; }

    /// <summary>
    /// GPT-2 style learned positional embeddings [MaxSeqLen, HiddenDim]. Present
    /// only when <see cref="ModelConfig.PositionalEncoding"/> is
    /// <see cref="Config.PositionalEncoding.Learned"/>; null for NoPE/RoPE/ALiBi.
    /// </summary>
    public Tensor<float>? PositionEmbedding { get; }

    // Raw quantized data for non-block tensors (embedding, lm_head)
    public byte[]? RawEmbedding { get; set; }
    public QuantDType? RawEmbeddingDtype { get; set; }
    public byte[]? RawLmHead { get; set; }
    public QuantDType? RawLmHeadDtype { get; set; }

    // Per-block weights (Attention, FFN, Norms)
    public BlockWeights[] Blocks { get; }

    // GGUF metadata
    public Format.ModelMetaData? GgufMeta { get; set; }
    public string? GgufPath { get; set; }
    public bool IsMoE { get; set; }

    // The loader used during InitializeWeights
    protected IModelLoader? Loader { get; }

    protected TransformerWeights(
        ModelConfig config,
        Tensor<float> embedding,
        Tensor<float>? lmHead,
        Tensor<float> finalNormW,
        Tensor<float>? finalNormB,
        BlockWeights[] blocks,
        IModelLoader? loader,
        Tensor<float>? positionEmbedding = null)
    {
        Config = config;
        EmbeddingWeight = embedding;
        LmHeadWeight = lmHead;
        FinalNormWeight = finalNormW;
        FinalNormBias = finalNormB;
        Blocks = blocks;
        Loader = loader;
        PositionEmbedding = positionEmbedding;
    }

    /// <summary>Initialises weights using the stored <see cref="IModelLoader"/>.
    /// Called after construction — must be called exactly once.</summary>
    public abstract void InitializeWeights(IProgress<float>? progress = null);

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            EmbeddingWeight.Dispose();
            LmHeadWeight?.Dispose();
            FinalNormWeight.Dispose();
            FinalNormBias?.Dispose();
            PositionEmbedding?.Dispose();
            foreach (var block in Blocks) block.Dispose();
        }
    }

    public (Tensor<float>? target, BlockWeights? block, string? rawField) ResolveTarget(string name)
    {
        // The token embedding tensor is exactly "token_embd.weight". Exclude
        // "token_embd_norm.weight" (LFM2's pre-embedding RMSNorm of the token
        // embedding) — routing it here overwrote the real Q8_0 embedding raw
        // bytes with the tiny [hidden] F32 norm weights, which then blew up the
        // logits decode with an access violation. Also exclude "per_layer_token_embd"
        // (gemma-3n/gemma-4), a separate per-layer table.
        if (name.EndsWith(".weight", StringComparison.OrdinalIgnoreCase)
            && name.Contains("token_embd", StringComparison.OrdinalIgnoreCase)
            && !name.Contains("token_embd_norm", StringComparison.OrdinalIgnoreCase)
            && !name.Contains("per_layer", StringComparison.OrdinalIgnoreCase))
            return (EmbeddingWeight, null, null);
        if (name.Contains("position_embd", StringComparison.OrdinalIgnoreCase)) return (PositionEmbedding, null, null);
        // LFM2's output RMSNorm is stored as "token_embd_norm.weight" (the gguf
        // converter maps LLM_TENSOR_OUTPUT_NORM_LFM2 "model.embedding_norm" to
        // that name). Route it to the final norm, NOT the embedding — the strict
        // token_embd match above already excludes it, so it must land here.
        if (name.Contains("output_norm", StringComparison.OrdinalIgnoreCase)
            || name.Contains("token_embd_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) return (FinalNormBias, null, null);
            return (FinalNormWeight, null, null);
        }
        if (name.Equals("output.weight", StringComparison.OrdinalIgnoreCase) || name.Equals("lm_head.weight", StringComparison.OrdinalIgnoreCase)) return (LmHeadWeight, null, null);

        var match = RegexGenerated.LayerIndexDotNDot.Match(name);
        if (match.Success && int.TryParse(match.Groups[1].Value, out int bIdx) && bIdx < Blocks.Length)
        {
            var block = Blocks[bIdx];
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase))
            {
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
            }
            else
            {
                if (name.Contains("post_attention_norm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("post_ffw_norm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                // Fused QKV (e.g. Phi-3: "blk.N.attn_qkv.weight") must be matched
                // BEFORE the individual attn_q/attn_k/attn_v substring checks,
                // because "attn_qkv".Contains("attn_q") is true.
                if (name.Contains("attn_qkv", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWqkv");
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWq");
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWk");
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWv");
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWo");
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return (null, block, null);

                if (IsMoE && name.Contains(".exps.", StringComparison.OrdinalIgnoreCase))
                {
                    var expMatch = RegexGenerated.ExpertIndex.Match(name);
                    if (expMatch.Success && int.TryParse(expMatch.Groups[1].Value, out int expIdx))
                    {
                        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                            return (null, block, $"RawWgateExp_{expIdx}");
                        if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                            return (null, block, $"RawWupExp_{expIdx}");
                        if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                            return (null, block, $"RawWdownExp_{expIdx}");
                    }
                    return (null, block, null);
                }

                if (IsMoE && name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                    return (null, block, "RawRouter");

                // LFM2 short-conv (no-attention) block weights. The conv kernel is
                // always stored F32 so it loads via the float path, not as a raw field.
                if (name.Contains("shortconv.in_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWScIn");
                if (name.Contains("shortconv.out_proj", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWScOut");
                // LFM2 short-conv depthwise kernel: F32, loaded via the float path
                // (ResolveFloatTarget maps it to WScConv). No raw field — it is always
                // dequantized into the float tensor, so match the block but no rawField.
                if (name.Contains("shortconv.conv.weight", StringComparison.OrdinalIgnoreCase)) return (null, block, null);

                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWgate");
                if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWup");
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return (null, block, "RawWf2");
            }
        }
        return (null, null, null);
    }

    public Tensor<float>? ResolveFloatTarget(string name)
    {
        if (name.EndsWith(".weight", StringComparison.OrdinalIgnoreCase)
            && name.Contains("token_embd", StringComparison.OrdinalIgnoreCase)
            && !name.Contains("token_embd_norm", StringComparison.OrdinalIgnoreCase)
            && !name.Contains("per_layer", StringComparison.OrdinalIgnoreCase)) return EmbeddingWeight;
        if (name.Contains("position_embd", StringComparison.OrdinalIgnoreCase)) return PositionEmbedding;
        if (name.Contains("output_norm", StringComparison.OrdinalIgnoreCase)
            || name.Contains("token_embd_norm", StringComparison.OrdinalIgnoreCase))
        {
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase)) return FinalNormBias;
            return FinalNormWeight;
        }
        if (name.Equals("output.weight", StringComparison.OrdinalIgnoreCase) || name.Equals("lm_head.weight", StringComparison.OrdinalIgnoreCase)) return LmHeadWeight;

        var match = RegexGenerated.LayerIndexDotNDot.Match(name);
        if (match.Success && int.TryParse(match.Groups[1].Value, out int bIdx) && bIdx < Blocks.Length)
        {
            var b = Blocks[bIdx];
            if (name.Contains("bias", StringComparison.OrdinalIgnoreCase))
            {
                if (IsMoE && name.Contains(".exps.", StringComparison.OrdinalIgnoreCase))
                {
                    var expMatch = RegexGenerated.ExpertIndex.Match(name);
                    if (expMatch.Success && int.TryParse(expMatch.Groups[1].Value, out int expIdx))
                    {
                        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WgateExpBias, expIdx, () => new Tensor<float>(Config.FfnDim), v => b.WgateExpBias = v);
                        if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WupExpBias, expIdx, () => new Tensor<float>(Config.FfnDim), v => b.WupExpBias = v);
                        if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WdownExpBias, expIdx, () => new Tensor<float>(Config.HiddenDim), v => b.WdownExpBias = v);
                    }
                    return null;
                }
                if (IsMoE && name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                    return b.WRouterBias ??= new Tensor<float>(Config.NumExperts);
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return b.WqBias;
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return b.WkBias;
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return b.WvBias;
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return b.WoBias;
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return b.Norm1B;
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return b.Norm2B;
                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase)) return b.Wf1Bias;
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return b.Wf2Bias;
            }
else
            {
                // LFM2 short-conv (no-attention) block weights
                if (name.Contains("shortconv.in_proj", StringComparison.OrdinalIgnoreCase))
                    return b.WScIn ??= new Tensor<float>(Config.HiddenDim, 3 * Config.HiddenDim);
                if (name.Contains("shortconv.out_proj", StringComparison.OrdinalIgnoreCase))
                    return b.WScOut ??= new Tensor<float>(Config.HiddenDim, Config.HiddenDim);
                if (name.Contains("shortconv.conv.weight", StringComparison.OrdinalIgnoreCase))
                    return b.WScConv ??= new Tensor<float>(Config.ShortConvCacheLength, Config.HiddenDim);

                if (name.Contains("post_attention_norm", StringComparison.OrdinalIgnoreCase))
                {
                    b.PostNorm1W ??= new Tensor<float>(Config.HiddenDim);
                    return b.PostNorm1W;
                }
                if (name.Contains("post_ffw_norm", StringComparison.OrdinalIgnoreCase))
                {
                    b.PostNorm2W ??= new Tensor<float>(Config.HiddenDim);
                    return b.PostNorm2W;
                }
                if (name.Contains("attn_q_norm", StringComparison.OrdinalIgnoreCase))
                {
                    b.QNormW ??= new Tensor<float>(Config.HeadDim);
                    return b.QNormW;
                }
                if (name.Contains("attn_k_norm", StringComparison.OrdinalIgnoreCase))
                {
                    b.KNormW ??= new Tensor<float>(Config.HeadDim);
                    return b.KNormW;
                }
                if (name.Contains("attn_q", StringComparison.OrdinalIgnoreCase) || name.Contains("q_proj", StringComparison.OrdinalIgnoreCase)) return b.Wq;
                if (name.Contains("attn_k", StringComparison.OrdinalIgnoreCase) || name.Contains("k_proj", StringComparison.OrdinalIgnoreCase)) return b.Wk;
                if (name.Contains("attn_v", StringComparison.OrdinalIgnoreCase) || name.Contains("v_proj", StringComparison.OrdinalIgnoreCase)) return b.Wv;
                if (name.Contains("attn_output", StringComparison.OrdinalIgnoreCase) || name.Contains("o_proj", StringComparison.OrdinalIgnoreCase)) return b.Wo;
                if (name.Contains("attn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("input_layernorm", StringComparison.OrdinalIgnoreCase)) return b.Norm1W;
                if (name.Contains("ffn_norm", StringComparison.OrdinalIgnoreCase) || name.Contains("post_attention_layernorm", StringComparison.OrdinalIgnoreCase)) return b.Norm2W;

                if (IsMoE && name.Contains(".exps.", StringComparison.OrdinalIgnoreCase))
                {
                    var expMatch = RegexGenerated.ExpertIndex.Match(name);
                    if (expMatch.Success && int.TryParse(expMatch.Groups[1].Value, out int expIdx))
                    {
                        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WgateExp, expIdx, () => new Tensor<float>(Config.HiddenDim, Config.FfnDim), v => b.WgateExp = v);
                        if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WupExp, expIdx, () => new Tensor<float>(Config.HiddenDim, Config.FfnDim), v => b.WupExp = v);
                        if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WdownExp, expIdx, () => new Tensor<float>(Config.FfnDim, Config.HiddenDim), v => b.WdownExp = v);
                    }
                    return null;
                }

                if (IsMoE && name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                    return b.WRouter ??= new Tensor<float>(Config.HiddenDim, Config.NumExperts);

                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                {
                    b.Wf1 ??= new Tensor<float>(Config.HiddenDim, 2 * Config.FfnDim);
                    return b.Wf1;
                }
                if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase)) return b.Wf2;
            }
        }
        return null;
    }

    /// <summary>Gets or creates a per-expert float tensor in the shared dictionary.</summary>
    private static Tensor<float> GetOrAdd(
        Dictionary<int, Tensor<float>>? dict, int key,
        Func<Tensor<float>> factory,
        Action<Dictionary<int, Tensor<float>>> store)
    {
        if (dict is not null && dict.TryGetValue(key, out var existing))
            return existing;
        var value = factory();
        dict ??= [];
        dict[key] = value;
        store(dict);
        return value;
    }

    public static void SetRawField(BlockWeights block, string field, byte[] data, QuantDType dtype)
    {
        if (field.StartsWith("RawWgateExp_", StringComparison.Ordinal) &&
            int.TryParse(field.AsSpan(12), out int gateExp))
        {
            block.RawWgateExp ??= [];
            block.RawWgateExp[gateExp] = data;
            block.QuantDtypeWgateExp ??= [];
            block.QuantDtypeWgateExp[gateExp] = dtype;
            return;
        }
        if (field.StartsWith("RawWupExp_", StringComparison.Ordinal) &&
            int.TryParse(field.AsSpan(10), out int upExp))
        {
            block.RawWupExp ??= [];
            block.RawWupExp[upExp] = data;
            block.QuantDtypeWupExp ??= [];
            block.QuantDtypeWupExp[upExp] = dtype;
            return;
        }
        if (field.StartsWith("RawWdownExp_", StringComparison.Ordinal) &&
            int.TryParse(field.AsSpan(12), out int downExp))
        {
            block.RawWdownExp ??= [];
            block.RawWdownExp[downExp] = data;
            block.QuantDtypeWdownExp ??= [];
            block.QuantDtypeWdownExp[downExp] = dtype;
            return;
        }

        switch (field)
        {
            case "RawWq": block.RawWq = data; block.QuantDtypeWq = dtype; break;
            case "RawWk": block.RawWk = data; block.QuantDtypeWk = dtype; break;
            case "RawWv": block.RawWv = data; block.QuantDtypeWv = dtype; break;
            case "RawWo": block.RawWo = data; block.QuantDtypeWo = dtype; break;
            case "RawWgate": block.RawWgate = data; block.QuantDtypeWgate = dtype; break;
            case "RawWup": block.RawWup = data; block.QuantDtypeWup = dtype; break;
            case "RawWf1": block.RawWf1 = data; block.QuantDtypeWf1 = dtype; break;
            case "RawWf2": block.RawWf2 = data; block.QuantDtypeWf2 = dtype; break;
            case "RawWScIn": block.RawWScIn = data; block.QuantDtypeWScIn = dtype; break;
            case "RawWScOut": block.RawWScOut = data; block.QuantDtypeWScOut = dtype; break;
            case "RawRouter": block.RawRouter = data; block.QuantDtypeRouter = dtype; break;
        }
    }

    /// <summary>Records tensor metadata in the block's <see cref="BlockWeights.TensorMeta"/>
    /// dictionary (field → {offset, size, dtype}) without loading data.</summary>
    public static void SetTensorMeta(BlockWeights block, string field, long offset, int size, QuantDType dtype)
    {
        block.TensorMeta[field] = new TensorMeta(offset, size, dtype);
    }

    /// <summary>
    /// Returns the distinct quantization dtypes used across all weight tensors.
    /// A tensor stored as plain floats contributes <see cref="QuantDType.F32"/>;
    /// quantized tensors contribute their storage dtype. The result is sorted by
    /// enum value and omits nothing (an all-float model returns <c>[F32]</c>).
    /// Works for full, cached, and streaming weights alike: block dtype fields,
    /// expert dtype dictionaries, and the metadata-driven <see cref="BlockWeights.TensorMeta"/>
    /// are all consulted, so it remains correct even while layers are unloaded.
    /// </summary>
    public QuantDType[] GetUsedQuantizations()
    {
        var seen = new HashSet<QuantDType>();
        Add(seen, RawEmbeddingDtype, EmbeddingWeight is not null);
        Add(seen, RawLmHeadDtype, LmHeadWeight is not null);
        Add(seen, null, PositionEmbedding is not null);

        foreach (var block in Blocks)
        {
            // Per-tensor: prefer the recorded quant dtype; fall back to F32 when only floats are resident.
            Add(seen, block.QuantDtypeWq,   block.Wq   is not null);
            Add(seen, block.QuantDtypeWk,   block.Wk   is not null);
            Add(seen, block.QuantDtypeWv,   block.Wv   is not null);
            Add(seen, block.QuantDtypeWo,   block.Wo   is not null);
            Add(seen, block.QuantDtypeWgate, block.RawWgate is not null);
            Add(seen, block.QuantDtypeWup,  block.RawWup  is not null);
            Add(seen, block.QuantDtypeWf1,  block.Wf1  is not null);
            Add(seen, block.QuantDtypeWf2,  block.Wf2  is not null);
            Add(seen, block.QuantDtypeWScIn,  block.WScIn  is not null);
            Add(seen, block.QuantDtypeWScOut, block.WScOut is not null);
            Add(seen, null, block.WScConv is not null);
            Add(seen, block.QuantDtypeRouter, block.RawRouter is not null);

            if (block.QuantDtypeWgateExp is { } gateExp)
                foreach (var (_, d) in gateExp) seen.Add(d);
            if (block.QuantDtypeWupExp is { } upExp)
                foreach (var (_, d) in upExp) seen.Add(d);
            if (block.QuantDtypeWdownExp is { } downExp)
                foreach (var (_, d) in downExp) seen.Add(d);

            foreach (var meta in block.TensorMeta.Values)
                seen.Add(meta.Dtype);
        }

        return [.. seen.OrderBy(d => d)];
    }

    private static void Add(HashSet<QuantDType> seen, QuantDType? quant, bool hasFloat)
    {
        if (quant is { } q) seen.Add(q);
        else if (hasFloat) seen.Add(QuantDType.F32);
    }

    public sealed class BlockWeights : IDisposable
    {
        /// <summary>
        /// Zero-based index of this block within the model. Used by the
        /// attention layer to pick the per-layer RoPE base (sliding-window
        /// models such as Gemma-3 use a different theta for windowed layers).
        /// </summary>
        public int LayerIndex { get; init; }

        // Attention float tensors (nullable — Full mode populates all; Cached mode populates on demand)
        public Tensor<float>? Wq { get; set; }
        public Tensor<float>? Wk { get; set; }
        public Tensor<float>? Wv { get; set; }
        public Tensor<float>? Wo { get; set; }
        public Tensor<float>? WqBias { get; set; }
        public Tensor<float>? WkBias { get; set; }
        public Tensor<float>? WvBias { get; set; }
        public Tensor<float>? WoBias { get; set; }

        // FFN float tensors
        public Tensor<float>? Wf1 { get; set; }
        public Tensor<float>? Wf2 { get; set; }
        public Tensor<float>? Wf1Bias { get; set; }
        public Tensor<float>? Wf2Bias { get; set; }

        // Norm float tensors
        public Tensor<float>? Norm1W { get; set; }
        public Tensor<float>? Norm1B { get; set; }
        public Tensor<float>? Norm2W { get; set; }
        public Tensor<float>? Norm2B { get; set; }

        // Per-head Q/K normalization (Qwen3)
        public Tensor<float>? QNormW { get; set; }
        public Tensor<float>? KNormW { get; set; }

        // Post-attention and post-FFN norms (Gemma-3)
        public Tensor<float>? PostNorm1W { get; set; }
        public Tensor<float>? PostNorm2W { get; set; }

        // LFM2 short-conv (no-attention) float tensors
        public Tensor<float>? WScIn { get; set; }   // [HiddenDim, 3*HiddenDim]
        public Tensor<float>? WScOut { get; set; }  // [HiddenDim, HiddenDim]
        public Tensor<float>? WScConv { get; set; } // [ShortConvCacheLength, HiddenDim] (F32 conv kernel)

        // Quantized data (byte arrays)
        public byte[]? RawWq { get; set; }
        public byte[]? RawWk { get; set; }
        public byte[]? RawWv { get; set; }
        public byte[]? RawWo { get; set; }
        public byte[]? RawWgate { get; set; }
        public byte[]? RawWup { get; set; }
        public byte[]? RawWf1 { get; set; }
        public byte[]? RawWf2 { get; set; }

        // LFM2 short-conv (no-attention) quantized projections
        public byte[]? RawWScIn { get; set; }
        public byte[]? RawWScOut { get; set; }

        // MoE expert quantized data
        public Dictionary<int, byte[]>? RawWgateExp { get; set; }
        public Dictionary<int, byte[]>? RawWupExp { get; set; }
        public Dictionary<int, byte[]>? RawWdownExp { get; set; }
        public byte[]? RawRouter { get; set; }

        // MoE float tensors (populated for F32/F16 training exports, mirroring
        // the shared-tensor round trip used by dense/gated FFNs)
        public Tensor<float>? WRouter { get; set; }
        public Tensor<float>? WRouterBias { get; set; }
        public Dictionary<int, Tensor<float>>? WgateExp { get; set; }
        public Dictionary<int, Tensor<float>>? WgateExpBias { get; set; }
        public Dictionary<int, Tensor<float>>? WupExp { get; set; }
        public Dictionary<int, Tensor<float>>? WupExpBias { get; set; }
        public Dictionary<int, Tensor<float>>? WdownExp { get; set; }
        public Dictionary<int, Tensor<float>>? WdownExpBias { get; set; }

        // Per-tensor quantization dtype
        public QuantDType? QuantDtypeWq { get; set; }
        public QuantDType? QuantDtypeWk { get; set; }
        public QuantDType? QuantDtypeWv { get; set; }
        public QuantDType? QuantDtypeWo { get; set; }
        public QuantDType? QuantDtypeWgate { get; set; }
        public QuantDType? QuantDtypeWup { get; set; }
        public QuantDType? QuantDtypeWf1 { get; set; }
        public QuantDType? QuantDtypeWf2 { get; set; }
        public QuantDType? QuantDtypeWScIn { get; set; }
        public QuantDType? QuantDtypeWScOut { get; set; }
        public Dictionary<int, QuantDType>? QuantDtypeWgateExp { get; set; }
        public Dictionary<int, QuantDType>? QuantDtypeWupExp { get; set; }
        public Dictionary<int, QuantDType>? QuantDtypeWdownExp { get; set; }
        public QuantDType? QuantDtypeRouter { get; set; }

        // Tensor metadata (offset, size, dtype) populated by IModelLoader.PreInit
        public Dictionary<string, TensorMeta> TensorMeta { get; } = [];

        public BlockWeights() { }

        public BlockWeights(
            Tensor<float> wq, Tensor<float> wk, Tensor<float> wv, Tensor<float> wo,
            Tensor<float> wqB, Tensor<float> wkB, Tensor<float> wvB, Tensor<float> woB,
            Tensor<float> wf1, Tensor<float> wf2, Tensor<float> wf1B, Tensor<float> wf2B,
            Tensor<float> n1w, Tensor<float>? n1b, Tensor<float> n2w, Tensor<float>? n2b,
            Tensor<float>? qNorm, Tensor<float>? kNorm,
            Tensor<float>? postNorm1W = null, Tensor<float>? postNorm2W = null)
        {
            Wq = wq; Wk = wk; Wv = wv; Wo = wo;
            WqBias = wqB; WkBias = wkB; WvBias = wvB; WoBias = woB;
            Wf1 = wf1; Wf2 = wf2; Wf1Bias = wf1B; Wf2Bias = wf2B;
            Norm1W = n1w; Norm1B = n1b; Norm2W = n2w; Norm2B = n2b;
            QNormW = qNorm; KNormW = kNorm;
            PostNorm1W = postNorm1W; PostNorm2W = postNorm2W;
        }

        public void Dispose()
        {
            Wq?.Dispose(); Wk?.Dispose(); Wv?.Dispose(); Wo?.Dispose();
            WqBias?.Dispose(); WkBias?.Dispose(); WvBias?.Dispose(); WoBias?.Dispose();
            Wf1?.Dispose(); Wf2?.Dispose(); Wf1Bias?.Dispose(); Wf2Bias?.Dispose();
            Norm1W?.Dispose(); Norm1B?.Dispose(); Norm2W?.Dispose(); Norm2B?.Dispose();
            QNormW?.Dispose(); KNormW?.Dispose();
            PostNorm1W?.Dispose(); PostNorm2W?.Dispose();
            WScIn?.Dispose(); WScOut?.Dispose(); WScConv?.Dispose();
            WRouter?.Dispose(); WRouterBias?.Dispose();
            DisposeDict(WgateExp); DisposeDict(WgateExpBias);
            DisposeDict(WupExp); DisposeDict(WupExpBias);
            DisposeDict(WdownExp); DisposeDict(WdownExpBias);
        }

        private static void DisposeDict(Dictionary<int, Tensor<float>>? dict)
        {
            if (dict is null) return;
            foreach (var t in dict.Values) t.Dispose();
        }

        /// <summary>
        /// Disposes and nulls all float tensor and raw data references.
        /// Keeps <see cref="TensorMeta"/> intact for future reloading.
        /// </summary>
        public void ReleaseLayerData()
        {
            Dispose();
            Wq = null; Wk = null; Wv = null; Wo = null;
            WqBias = null; WkBias = null; WvBias = null; WoBias = null;
            Wf1 = null; Wf2 = null; Wf1Bias = null; Wf2Bias = null;
            Norm1W = null; Norm1B = null; Norm2W = null; Norm2B = null;
            QNormW = null; KNormW = null;
            PostNorm1W = null; PostNorm2W = null;
            WScIn = null; WScOut = null; WScConv = null;
            RawWq = null; RawWk = null; RawWv = null; RawWo = null;
            RawWgate = null; RawWup = null; RawWf1 = null; RawWf2 = null;
            RawWScIn = null; RawWScOut = null;
            QuantDtypeWq = null; QuantDtypeWk = null; QuantDtypeWv = null; QuantDtypeWo = null;
            QuantDtypeWgate = null; QuantDtypeWup = null; QuantDtypeWf1 = null; QuantDtypeWf2 = null;
            QuantDtypeWScIn = null; QuantDtypeWScOut = null;
            RawWgateExp = null; RawWupExp = null; RawWdownExp = null;
            QuantDtypeWgateExp = null; QuantDtypeWupExp = null; QuantDtypeWdownExp = null;
            RawRouter = null; QuantDtypeRouter = null;
            WRouter = null; WRouterBias = null;
            WgateExp = null; WgateExpBias = null;
            WupExp = null; WupExpBias = null;
            WdownExp = null; WdownExpBias = null;
        }
    }
}

/// <summary>Loads all weights into memory at once.</summary>
public sealed class TransformerWeightsFull(
    ModelConfig config,
    Tensor<float> embedding,
    Tensor<float>? lmHead,
    Tensor<float> finalNormW,
    Tensor<float>? finalNormB,
TransformerWeights.BlockWeights[] blocks,
    IModelLoader loader,
    Tensor<float>? positionEmbedding = null) : TransformerWeights(config, embedding, lmHead, finalNormW, finalNormB, blocks, loader, positionEmbedding)
{
    public override void InitializeWeights(IProgress<float>? progress = null) => Loader!.LoadAllWeights(this, progress);
}

/// <summary>
/// Streaming weights — loads and unloads one layer at a time during the forward pass.
/// Only a small window of layers has float tensors+raw data resident at any moment.
/// Each agent in LoadMode.Streaming requires its own instance.
/// </summary>
public sealed class TransformerWeightsStreaming(
    ModelConfig config,
    Tensor<float> embedding,
    Tensor<float>? lmHead,
    Tensor<float> finalNormW,
    Tensor<float>? finalNormB,
TransformerWeights.BlockWeights[] blocks,
    IModelLoader loader,
    Tensor<float>? positionEmbedding = null) : TransformerWeights(config, embedding, lmHead, finalNormW, finalNormB, blocks, loader, positionEmbedding)
{
    /// <summary>Reference to the TransformerBlock[] so loaded weights can be pushed via SetWeights.</summary>
    internal Layers.TransformerBlock[]? BlockRefs { get; set; }

    // Async preload tracking
    private Task? _preloadTask;
    private int _preloadLayerIndex = -1;
    private readonly Lock _preloadLock = new();

    /// <summary>Tracks which layers have been pushed into their TransformerBlock's LinearLayers.</summary>
    private readonly HashSet<int> _pushedLayers = [];

    /// <summary>
    /// Metadata-only initialisation — reads the GGUF header and populates
    /// <see cref="TransformerWeights.GgufMeta"/> and per-block
    /// <see cref="BlockWeights.TensorMeta"/> without loading any weight data.
    /// </summary>
    public override void InitializeWeights(IProgress<float>? progress = null)
    {
        var meta = Format.ModelFormatHelpers.LoadMetaForFile(GgufPath!);
        GgufMeta = meta;
        IsMoE = meta.Tensors.Any(t => t.Name.Contains(".exps."));

        // Populate TensorMeta for all blocks (file offsets, sizes, dtypes)
        foreach (var info in meta.Tensors)
        {
            var (target, block, rawField) = ResolveTarget(info.Name);
            if (block != null && rawField != null)
            {
                long rawSize = Core.Quantization.QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
                if (rawSize > 0)
                {
                    if (rawField == "RawWqkv")
                    {
                        // Fused QKV: register individual TensorMeta entries so the
                        // AttentionLayer constructor reads the correct dtype instead
                        // of defaulting to F32.
                        int partSize = (int)(rawSize / 3);
                        long baseOffset = meta.DataOffset + info.Offset;
                        SetTensorMeta(block, "RawWq", baseOffset, partSize, info.Dtype);
                        SetTensorMeta(block, "RawWk", baseOffset + partSize, partSize, info.Dtype);
                        SetTensorMeta(block, "RawWv", baseOffset + partSize * 2, partSize, info.Dtype);
                    }
                    else
                    {
                        SetTensorMeta(block, rawField, meta.DataOffset + info.Offset, (int)rawSize, info.Dtype);
                    }
                }
            }
        }

        _pushedLayers.Clear();
        progress?.Report(1f);

        // Load global non-block tensors (embedding, final norm, lm_head).
        // These are not per-layer and must be present before any forward pass.
        Loader!.LoadGlobalTensors(this);

        // Layer 0 async preload is deferred to CreateTransformer after
        // BlockRefs is set, to avoid racing with BuildBlock reading Blocks[0].
    }

    /// <summary>
    /// Ensures the given layer is loaded and pushed into its TransformerBlock.
    /// If an async preload is running for this layer, waits for it to complete first.
    /// Otherwise loads synchronously.
    /// </summary>
    public void EnsureLayerLoadedSync(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= Blocks.Length) return;

        bool needsPush = false;

        // For fused QKV models (e.g. Phi-3), Wq is never set — only RawWq is.
        // Check both so the layer isn't reloaded every forward pass.
        if (Blocks[layerIndex].Wq == null && Blocks[layerIndex].RawWq == null)
        {
            // Wait for any running preload of this layer
            lock (_preloadLock)
            {
                if (_preloadTask != null && _preloadLayerIndex == layerIndex)
                {
                    _preloadTask.GetAwaiter().GetResult();
                    _preloadTask = null;
                    _preloadLayerIndex = -1;
                }
            }

            if (Blocks[layerIndex].Wq == null && Blocks[layerIndex].RawWq == null)
            {
                Loader!.LoadLayerWeights(layerIndex, this);
            }
            needsPush = true;
        }

        if (needsPush || !_pushedLayers.Contains(layerIndex))
        {
            BlockRefs?[layerIndex]?.SetWeights(Blocks[layerIndex]);
            _pushedLayers.Add(layerIndex);
        }
    }

    /// <summary>
    /// Fires a background task to load the given layer's raw+float data.
    /// Does NOT push into the TransformerBlock — that happens on the forward
    /// thread when <see cref="EnsureLayerLoadedSync"/> is called for this layer.
    /// </summary>
    public void PreloadLayerAsync(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= Blocks.Length) return;
        if (Blocks[layerIndex].Wq != null || Blocks[layerIndex].RawWq != null) return;

        lock (_preloadLock)
        {
            _preloadTask = Task.Run(() =>
            {
                Loader!.LoadLayerWeights(layerIndex, this);
            });
            _preloadLayerIndex = layerIndex;
        }
    }

    /// <summary>
    /// Frees all weight data (float tensors + raw quantized data) for a layer.
    /// The layer's <see cref="BlockWeights.TensorMeta"/> is preserved so it
    /// can be reloaded later.
    /// </summary>
    public void FreeLayer(int layerIndex)
    {
        if (layerIndex < 0 || layerIndex >= Blocks.Length) return;
        if (Blocks[layerIndex].Wq == null && Blocks[layerIndex].RawWq == null) return;

        // Push empty weights into LinearLayers (clears raw data references)
        BlockRefs?[layerIndex]?.SetWeights(new BlockWeights());

        // Free float tensors in LinearLayers (disposes pre-allocated tensors)
        BlockRefs?[layerIndex]?.FreeFloatWeights();

        // Dispose float tensors and null all fields in BlockWeights; keeps TensorMeta
        Blocks[layerIndex].ReleaseLayerData();

        _pushedLayers.Remove(layerIndex);
    }

    /// <summary>
    /// Cleans up after a forward pass by freeing any remaining loaded layers.
    /// Called by <see cref="Transformer.Forward"/> after the architecture forward
    /// completes. Without this, the last layer(s) would remain loaded across
    /// consecutive forward passes (token generation), gradually consuming memory.
    /// </summary>
    public void CompleteForward()
    {
        // Free any layers that are still loaded (typical: last layer(s) of the pass)
        for (int i = 0; i < Blocks.Length; i++)
        {
            if (Blocks[i].Wq != null || Blocks[i].RawWq != null)
                FreeLayer(i);
        }
    }
}


