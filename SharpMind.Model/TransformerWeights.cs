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
    // Lazy's default mode is thread-safe: a GPU engine and a Medusa generator can ask at the same time.
    private Lazy<Tensor<float>>? _lmHead;
    // The factory a lazily loaded head rebuilds itself from, retained so
    // DisposeMaterializedLmHead can re-arm the lazy instead of leaving a disposed tensor.
    private Func<Tensor<float>>? _lmHeadFactory;

    /// <summary>
    /// The untied output head as floats, or null for a tied model. A head loaded from a file is
    /// dequantized from <see cref="RawLmHead"/> on first access: the CPU projection reads the raw
    /// bytes, so only float consumers (Medusa, a GPU head without an on-device kernel, training
    /// export) pay for this copy, and which of them runs is not known at load.
    /// </summary>
    public Tensor<float>? LmHeadWeight => _lmHead?.Value;

    /// <summary>True when the model has its own output head, whether or not its floats exist yet.</summary>
    public bool HasLmHead => _lmHead is not null;

    public void SetLmHead(Tensor<float> head)
    {
        _lmHeadFactory = null;
        _lmHead = new Lazy<Tensor<float>>(head);
    }

    /// <summary>Registers an output head whose floats <paramref name="dequantize"/> builds on first access.</summary>
    internal void SetLazyLmHead(Func<Tensor<float>> dequantize)
    {
        _lmHeadFactory = dequantize;
        _lmHead = new Lazy<Tensor<float>>(dequantize);
    }
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

    /// <summary>
    /// Allocates the raw quantized buffer a block tensor is read into. The base returns a
    /// fresh array; <see cref="TransformerWeightsStreaming"/> overrides this with a reused
    /// <see cref="Core.Memory.ByteBufferPool"/> buffer so the layer rotation stays
    /// allocation-free. Top-level (embedding/head) tensors keep fresh arrays — they are
    /// loaded once and never returned. MoE expert residency will extend the same seam.
    /// </summary>
    public virtual byte[] AllocateRawBuffer(int byteCount) => new byte[byteCount];

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
        _lmHead = lmHead is null ? null : new Lazy<Tensor<float>>(lmHead);
        FinalNormWeight = finalNormW;
        FinalNormBias = finalNormB;
        Blocks = blocks;
        Loader = loader;
        PositionEmbedding = positionEmbedding;
    }

    /// <summary>Initialises weights using the stored <see cref="IModelLoader"/>.
    /// Called after construction — must be called exactly once.</summary>
    public abstract void InitializeWeights(IProgress<float>? progress = null, CancellationToken? cancellationToken = null);

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
            if (_lmHead is { IsValueCreated: true }) _lmHead.Value.Dispose();
            _lmHead = null;
            _lmHeadFactory = null;
            FinalNormWeight.Dispose();
            FinalNormBias?.Dispose();
            PositionEmbedding?.Dispose();
            foreach (var block in Blocks) block.Dispose();
            // Drop the raw quantized byte[] references so the GC can reclaim
            // them the moment the owning graph is unreachable. They are LOH
            // data (one model's quantized weights can be hundreds of MB to
            // gigabytes); leaving them referenced until a rare compaction
            // ratchets process memory by ~one model footprint per unload.
            RawEmbedding = null;
            RawLmHead = null;
        }
    }

    /// <summary>
    /// True for the untied output-head tensor names (<c>output.weight</c>,
    /// <c>lm_head.weight</c>) outside a block. Must match the routing in
    /// <see cref="ResolveTarget"/>; the loaders use it to send the head through
    /// <c>LoadSingleTensor</c> even when <see cref="LmHeadWeight"/> is still null
    /// (streaming loads create it lazily on first access).
    /// </summary>
    public static bool IsLmHeadTensorName(string name)
        => name.Equals("output.weight", StringComparison.OrdinalIgnoreCase)
           || name.Equals("lm_head.weight", StringComparison.OrdinalIgnoreCase);

    /// <summary>True when the lazy untied head has already been materialized.</summary>
    public bool LmHeadIsMaterialized => _lmHead is { IsValueCreated: true };

    /// <summary>
    /// Frees the float head <em>without</em> materializing it. Accessing
    /// <see cref="LmHeadWeight"/> would dequantize a Q8_0 head (~544 MB for
    /// Qwen2.5-1.5B) just to dispose it; callers that only need the raw bytes
    /// for the quantized projection should use this to release a head a consumer
    /// already forced, and otherwise leave it un-materialized. A head registered
    /// through <see cref="SetLazyLmHead"/> is re-armed from its raw bytes, so a
    /// later <see cref="LmHeadWeight"/> re-materializes it instead of handing out
    /// the disposed tensor.
    /// </summary>
    public void DisposeMaterializedLmHead()
    {
        if (_lmHead is not { IsValueCreated: true }) return;
        _lmHead.Value.Dispose();
        _lmHead = _lmHeadFactory is null ? _lmHead : new Lazy<Tensor<float>>(_lmHeadFactory);
    }

    /// <summary>
    /// Single source of truth for "does this GGUF describe a mixture-of-experts model?"
    /// </summary>
    /// <remarks>
    /// Every caller must use this, not a local subset of the patterns. The
    /// streaming path once used only <c>".exps."</c>, which misses qwen2moe's
    /// <c>ffn_gate_exps.weight</c> (no leading dot). That left
    /// <see cref="IsMoE"/> false there while the layers were still built as MoE
    /// from the architecture string, so every MoE tensor fell through to the
    /// dense ffn_gate branch: the router ended up with no weights and the
    /// forward threw "RawQuantizedData is null and no float fallback available".
    /// </remarks>
    public static bool IsMoEGguf(IEnumerable<TensorInfo> tensors) =>
        tensors.Any(t =>
            t.Name.Contains(".exps.", StringComparison.OrdinalIgnoreCase) ||
            t.Name.Contains("_exps", StringComparison.OrdinalIgnoreCase) ||
            t.Name.Contains("ffn_gate_inp", StringComparison.OrdinalIgnoreCase) ||
            (t.Name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(t.Name)));

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

                // MoE shared expert (Qwen1.5-MoE). Matched BEFORE the routed-expert
                // branch because "ffn_gate_shexp" also contains "ffn_gate" — routing
                // it as an expert or as the router would bind it to the wrong weight.
                if (IsMoE && name.Contains("_shexp", StringComparison.OrdinalIgnoreCase))
                {
                    // ffn_gate_inp_shexp is the F32 sigmoid gate row; let the float
                    // path fill SharedGateInp rather than short-circuiting on raw bytes.
                    if (name.Contains("gate_inp", StringComparison.OrdinalIgnoreCase))
                        return (null, block, null);
                    if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                        return (null, block, "RawWSharedGate");
                    if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                        return (null, block, "RawWSharedUp");
                    if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                        return (null, block, "RawWSharedDown");
                    return (null, block, null);
                }

                bool isMoEExpert = IsMoE && (
                    name.Contains(".exps.", StringComparison.OrdinalIgnoreCase) ||
                    name.Contains("_exps", StringComparison.OrdinalIgnoreCase) ||
                    (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)) ||
                    (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)) ||
                    (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)));
                if (isMoEExpert)
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
                    // Fused expert stack: "ffn_gate_exps"/"ffn_up_exps"/"ffn_down_exps"
                    // with no {e} index in the name. The loader splits the 3D tensor.
                    if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                        return (null, block, "RawWgateExpFused");
                    if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                        return (null, block, "RawWupExpFused");
                    if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                        return (null, block, "RawWdownExpFused");
                    return (null, block, null);
                }

                if (IsMoE && name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase)
                    && !name.Contains("_shexp", StringComparison.OrdinalIgnoreCase))
                    // Routers are F32 in practice (llama.cpp always writes them unquantized),
                    // so no raw field is returned: that keeps the loader on the float path,
                    // which is what fills WRouter. Returning a raw field here made the loader
                    // store the bytes and return early, leaving the router with no weights —
                    // "RawQuantizedData is null and no float fallback available" at first
                    // forward. Without this guard the name fell through to the dense
                    // ffn_gate branch below and was bound to the gated FFN's RawWgate.
                    return (null, block, null);

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
                bool isMoEBiasExp = IsMoE && (
                    name.Contains(".exps.", StringComparison.OrdinalIgnoreCase) ||
                    name.Contains("_exps", StringComparison.OrdinalIgnoreCase) ||
                    (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)) ||
                    (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)) ||
                    (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)));
                if (isMoEBiasExp)
                {
                    var expMatch = RegexGenerated.ExpertIndex.Match(name);
                    if (expMatch.Success && int.TryParse(expMatch.Groups[1].Value, out int expIdx))
                    {
                        int expFfn = Config.ResolvedExpertFfnDim;
                        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WgateExpBias, expIdx, () => new Tensor<float>(expFfn), v => b.WgateExpBias = v);
                        if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WupExpBias, expIdx, () => new Tensor<float>(expFfn), v => b.WupExpBias = v);
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

                // MoE shared expert. "_shexp" is matched on its own, never gated on
                // HasSharedExpert: if the file carries *_shexp tensors but its metadata
                // lacks expert_shared_feed_forward_length, letting these fall through
                // sent them to the router branch below, where ffn_gate_inp_shexp (rank 1,
                // [HiddenDim]) overwrote the real [HiddenDim, NumExperts] router. The
                // symptom was the router reporting "weightElements=2048, expected=122880"
                // at the first forward, and which tensor won depended on file order.
                if (IsMoE && name.Contains("_shexp", StringComparison.OrdinalIgnoreCase))
                {
                    if (name.Contains("gate_inp", StringComparison.OrdinalIgnoreCase))
                        return b.SharedGateInp ??= new Tensor<float>(Config.HiddenDim);
                    int sharedFfn = Config.ResolvedSharedFfnDim;
                    if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                        return b.WSharedGate ??= new Tensor<float>(Config.HiddenDim, sharedFfn);
                    if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                        return b.WSharedUp ??= new Tensor<float>(Config.HiddenDim, sharedFfn);
                    if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                        return b.WSharedDown ??= new Tensor<float>(sharedFfn, Config.HiddenDim);
                    return null;
                }

                bool isMoEExpW = IsMoE && (
                    name.Contains(".exps.", StringComparison.OrdinalIgnoreCase) ||
                    name.Contains("_exps", StringComparison.OrdinalIgnoreCase) ||
                    (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)) ||
                    (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)) ||
                    (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase) && RegexGenerated.ExpertIndex.IsMatch(name)));
                if (isMoEExpW)
                {
                    var expMatch = RegexGenerated.ExpertIndex.Match(name);
                    if (expMatch.Success && int.TryParse(expMatch.Groups[1].Value, out int expIdx))
                    {
                        // Routed experts use the expert FFN width, which is narrower
                        // than Config.FfnDim on models that also have a shared expert.
                        int expFfn = Config.ResolvedExpertFfnDim;
                        if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WgateExp, expIdx, () => new Tensor<float>(Config.HiddenDim, expFfn), v => b.WgateExp = v);
                        if (name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WupExp, expIdx, () => new Tensor<float>(Config.HiddenDim, expFfn), v => b.WupExp = v);
                        if (name.Contains("ffn_down", StringComparison.OrdinalIgnoreCase))
                            return GetOrAdd(b.WdownExp, expIdx, () => new Tensor<float>(expFfn, Config.HiddenDim), v => b.WdownExp = v);
                    }
                    return null;
                }

                // The router. "_shexp" is excluded explicitly: ffn_gate_inp_shexp also
                // contains "ffn_gate", and WRouter is ??=, so an unfiltered match lets
                // the shared gate silently replace the real router.
                if (IsMoE
                    && name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase)
                    && !name.Contains("_shexp", StringComparison.OrdinalIgnoreCase))
                    return b.WRouter ??= new Tensor<float>(Config.HiddenDim, Config.NumExperts);

                if (name.Contains("ffn_gate", StringComparison.OrdinalIgnoreCase) || name.Contains("ffn_up", StringComparison.OrdinalIgnoreCase))
                {
                    // Full float loads allocate Wf1 up front. Inference loads (quantized-resident,
                    // streaming) only need it when gate/up arrive without raw bytes: both loaders store
                    // a tensor's raw bytes before resolving its float target, and a block holding them
                    // runs its gated layer from those and never reads the floats.
                    if (b.Wf1 is null && (b.RawWgate is not null || b.RawWup is not null))
                        return null;
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
            case "RawWSharedGate": block.RawWSharedGate = data; block.QuantDtypeSharedGate = dtype; break;
            case "RawWSharedUp": block.RawWSharedUp = data; block.QuantDtypeSharedUp = dtype; break;
            case "RawWSharedDown": block.RawWSharedDown = data; block.QuantDtypeSharedDown = dtype; break;
            case "RawWSharedGateInp": block.RawWSharedGateInp = data; break;
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
        Add(seen, RawLmHeadDtype, HasLmHead);
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
            Add(seen, block.QuantDtypeSharedGate, block.RawWSharedGate is not null);
            Add(seen, block.QuantDtypeSharedUp, block.RawWSharedUp is not null);
            Add(seen, block.QuantDtypeSharedDown, block.RawWSharedDown is not null);

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

        // MoE shared expert (Qwen1.5-MoE "*_shexp" tensors). These run on every
        // token alongside the top-k routed experts.
        public byte[]? RawWSharedGate { get; set; }
        public byte[]? RawWSharedUp { get; set; }
        public byte[]? RawWSharedDown { get; set; }
        public byte[]? RawWSharedGateInp { get; set; }
        public Tensor<float>? WSharedGate { get; set; }
        public Tensor<float>? WSharedGateBias { get; set; }
        public Tensor<float>? WSharedUp { get; set; }
        public Tensor<float>? WSharedUpBias { get; set; }
        public Tensor<float>? WSharedDown { get; set; }
        public Tensor<float>? WSharedDownBias { get; set; }
        public Tensor<float>? SharedGateInp { get; set; }

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
        public QuantDType? QuantDtypeSharedGate { get; set; }
        public QuantDType? QuantDtypeSharedUp { get; set; }
        public QuantDType? QuantDtypeSharedDown { get; set; }

        // Tensor metadata (offset, size, dtype) populated by IModelLoader.PreInit
        public Dictionary<string, TensorMeta> TensorMeta { get; } = [];

        /// <summary>Allocates pooled raw buffers on behalf of this block; wired by the
        /// streaming weights so layer-derived buffers (the fused gate+up FFN payload, MoE
        /// expert planes) also round-trip through the reuse pool. Null in full-resident mode,
        /// where weights load once and there is nothing to recycle.</summary>
        internal Func<int, byte[]>? BufferAllocator { get; set; }

        /// <summary>Raw buffers a layer derived from pooled inputs and registered so
        /// <see cref="FreeLayer"/> can hand them back to the pool instead of the GC.</summary>
        private List<byte[]>? _ownedPooled;

        /// <summary>
        /// Borrows a raw buffer for a layer-derived value (the GatedFfn fused gate+up array).
        /// When the streaming pool is wired the buffer is registered as owned so
        /// <see cref="CaptureOwnedBuffers"/> returns it for pooling at unload; otherwise it is
        /// a plain allocation (full-resident managers load once per run).
        /// </summary>
        internal byte[] AllocateBlockBuffer(int size)
        {
            if (BufferAllocator is null) return new byte[size];
            byte[] buf = BufferAllocator(size);
            _ownedPooled ??= [];
            _ownedPooled.Add(buf);
            return buf;
        }

        /// <summary>Returns the registered layer-derived buffers and detaches them so an
        /// unload can return them to the pool.</summary>
        internal List<byte[]>? CaptureOwnedBuffers()
        {
            var owned = _ownedPooled;
            _ownedPooled = null;
            return owned;
        }

        /// <summary>Total bytes of raw + derived buffers currently attached to this block.
        /// Streaming uses it to size the pool budget to the resident working set.</summary>
        internal long AttachedRawBytes()
        {
            long sum = 0;
            foreach (byte[] b in CaptureRawBuffers()) sum += b.Length;
            if (_ownedPooled is not null)
                foreach (byte[] b in _ownedPooled) sum += b.Length;
            return sum;
        }

        public BlockWeights() { }

        /// <summary>
        /// Snapshots every raw quantized byte[] currently attached to this block (attention,
        /// FFN, short-conv, router, shared expert, per-expert dictionaries). Streaming uses
        /// this before <see cref="ReleaseLayerData"/> nulls the fields so the arrays can be
        /// returned to the buffer pool instead of dropped for the GC.
        /// </summary>
        public IEnumerable<byte[]> CaptureRawBuffers()
        {
            if (RawWq is not null) yield return RawWq;
            if (RawWk is not null) yield return RawWk;
            if (RawWv is not null) yield return RawWv;
            if (RawWo is not null) yield return RawWo;
            if (RawWgate is not null) yield return RawWgate;
            if (RawWup is not null) yield return RawWup;
            if (RawWf1 is not null) yield return RawWf1;
            if (RawWf2 is not null) yield return RawWf2;
            if (RawWScIn is not null) yield return RawWScIn;
            if (RawWScOut is not null) yield return RawWScOut;
            if (RawRouter is not null) yield return RawRouter;
            if (RawWSharedGate is not null) yield return RawWSharedGate;
            if (RawWSharedUp is not null) yield return RawWSharedUp;
            if (RawWSharedDown is not null) yield return RawWSharedDown;
            if (RawWSharedGateInp is not null) yield return RawWSharedGateInp;
            if (RawWgateExp is not null)
                foreach (var buf in RawWgateExp.Values) if (buf is not null) yield return buf;
            if (RawWupExp is not null)
                foreach (var buf in RawWupExp.Values) if (buf is not null) yield return buf;
            if (RawWdownExp is not null)
                foreach (var buf in RawWdownExp.Values) if (buf is not null) yield return buf;
        }

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
            WSharedGate?.Dispose(); WSharedGateBias?.Dispose();
            WSharedUp?.Dispose(); WSharedUpBias?.Dispose();
            WSharedDown?.Dispose(); WSharedDownBias?.Dispose();
            SharedGateInp?.Dispose();
            // Null the raw quantized byte[] references too — same rationale as
            // TransformerWeights.Dispose: these can be a model's entire weight
            // footprint in LOH data, and leaving the references rooted in the
            // (now-unreachable) blocks delays reclamation until a compaction.
            RawWq = null; RawWk = null; RawWv = null; RawWo = null;
            RawWgate = null; RawWup = null; RawWf1 = null; RawWf2 = null;
            RawWScIn = null; RawWScOut = null;
            RawRouter = null;
            RawWgateExp = null; RawWupExp = null; RawWdownExp = null;
            RawWSharedGate = null; RawWSharedUp = null; RawWSharedDown = null; RawWSharedGateInp = null;
            _ownedPooled = null;
        }

        private static void DisposeDict(Dictionary<int, Tensor<float>>? dict)
        {
            if (dict is null) return;
            foreach (var t in dict.Values) t.Dispose();
        }

        /// <summary>
        /// Releases the large per-layer payloads for streaming (the 2D float
        /// weights and the raw quantized bytes), keeping the small tensors the
        /// built layers hold by reference (norms and biases) and
        /// <see cref="TensorMeta"/> for reloading. <see cref="Dispose"/> frees
        /// everything on final teardown.
        /// </summary>
        public void ReleaseLayerData()
        {
            // Free only the large payloads: the 2D float weights and the raw
            // quantized bytes. Those are what streaming exists to unload.
            Wq?.Dispose(); Wq = null;
            Wk?.Dispose(); Wk = null;
            Wv?.Dispose(); Wv = null;
            Wo?.Dispose(); Wo = null;
            Wf1?.Dispose(); Wf1 = null;
            Wf2?.Dispose(); Wf2 = null;
            WScIn?.Dispose(); WScIn = null;
            WScOut?.Dispose(); WScOut = null;
            WRouter?.Dispose(); WRouter = null;
            DisposeDict(WgateExp); WgateExp = null;
            DisposeDict(WupExp); WupExp = null;
            DisposeDict(WdownExp); WdownExp = null;

            RawWq = null; RawWk = null; RawWv = null; RawWo = null;
            RawWgate = null; RawWup = null; RawWf1 = null; RawWf2 = null;
            RawWScIn = null; RawWScOut = null;
            QuantDtypeWq = null; QuantDtypeWk = null; QuantDtypeWv = null; QuantDtypeWo = null;
            QuantDtypeWgate = null; QuantDtypeWup = null; QuantDtypeWf1 = null; QuantDtypeWf2 = null;
            QuantDtypeWScIn = null; QuantDtypeWScOut = null;
            RawWgateExp = null; RawWupExp = null; RawWdownExp = null;
            QuantDtypeWgateExp = null; QuantDtypeWupExp = null; QuantDtypeWdownExp = null;
            RawRouter = null; QuantDtypeRouter = null;
            WSharedGate?.Dispose(); WSharedGate = null;
            WSharedUp?.Dispose(); WSharedUp = null;
            WSharedDown?.Dispose(); WSharedDown = null;
            RawWSharedGate = null; RawWSharedUp = null; RawWSharedDown = null; RawWSharedGateInp = null;
            QuantDtypeSharedGate = null; QuantDtypeSharedUp = null; QuantDtypeSharedDown = null;

            // Deliberately NOT disposed/nulled: the small tensors the built layers
            // capture by reference for their whole lifetime. NormLayer stores the
            // norm tensor it was handed at construction and TransformerBlock never
            // repoints it (it only copies data into it), and LinearLayer stores the
            // bias tensor directly, so disposing these here left a disposed
            // NativeBuffer in the forward path and the next forward threw
            // ObjectDisposedException on the norm's first read. They are also tiny:
            // Norm1W/Norm1B/Norm2W/Norm2B, QNormW/KNormW, PostNorm1W/PostNorm2W,
            // WqBias/WkBias/WvBias/WoBias, Wf1Bias/Wf2Bias, WRouterBias, the
            // per-expert biases and the F32 short-conv kernel (WScConv).
            // Dispose() still frees them all on final teardown.
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
    public override void InitializeWeights(IProgress<float>? progress = null, CancellationToken? cancellationToken = null)
        => Loader!.LoadAllWeights(this, progress, cancellationToken);
}

/// <summary>
/// Streaming weights — loads and unloads layers as the forward pass advances, keeping a
/// small window resident. <paramref name="positionEmbedding"/>-scale memory stays bounded:
/// as the block loop reaches layer i it unloads the layer <see cref="ResidentWindow"/>
/// behind and asynchronously preloads the layer that far ahead, overlapping I/O with compute.
/// Each agent in LoadMode.Streaming requires its own instance.
/// </summary>
public sealed class TransformerWeightsStreaming : TransformerWeights
{
    public TransformerWeightsStreaming(
        ModelConfig config,
        Tensor<float> embedding,
        Tensor<float>? lmHead,
        Tensor<float> finalNormW,
        Tensor<float>? finalNormB,
        TransformerWeights.BlockWeights[] blocks,
        IModelLoader loader,
        Tensor<float>? positionEmbedding = null)
        : base(config, embedding, lmHead, finalNormW, finalNormB, blocks, loader, positionEmbedding)
    {
        // Layer-derived buffers (fused gate+up, MoE expert planes) acquire through the same
        // pooling, sized by the block allocator, so unloads round-trip everything.
        foreach (BlockWeights b in blocks)
            b.BufferAllocator = AllocateRawBuffer;
    }
    /// <summary>
    /// Layers kept resident on either side of the block loop's current position. 0 would
    /// unload the current layer before its successors finish and is not valid; the default is
    /// the historical stride of one (unload one behind, preload one ahead), so ≈3 layers are
    /// resident at any moment. Larger values keep more layers hot between tokens in exchange
    /// for memory. The pool retention budget follows it: enough to cover the layers in flight
    /// plus the layer the loop is loading next, so the hot reload cycle never reallocates.
    /// </summary>
    private int _residentWindow = 1;
    public int ResidentWindow
    {
        get => _residentWindow;
        set
        {
            _residentWindow = Math.Max(0, value);
            UpdatePoolBudget();
        }
    }

    internal void UpdatePoolBudget()
    {
        if (_peakLayerBytes > 0)
            _rawBuffers.MaxRetainedBytes = (long)(_residentWindow + 2) * _peakLayerBytes;
    }

    /// <summary>Largest single-layer raw+derived footprint seen; the pool budget is
    /// <c>(ResidentWindow + 2) × this</c> so a free and its next acquire never cross an
    /// eviction in steady state.</summary>
    private long _peakLayerBytes;

    /// <summary>Reference to the TransformerBlock[] so loaded weights can be pushed via SetWeights.</summary>
    internal Layers.TransformerBlock[]? BlockRefs { get; set; }

    private readonly Core.Memory.ByteBufferPool _rawBuffers = new();

    /// <summary>
    /// Borrows a raw quantized buffer for a block tensor from the reuse pool (bounded by
    /// <see cref="ResidentWindow"/>). The pool grows its budget with the largest tensor seen
    /// times the resident window, so a sliding decode keeps allocations at zero in steady state.
    /// </summary>
    public override byte[] AllocateRawBuffer(int byteCount)
    {
        _rawBuffers.ResidentLayers = ResidentWindow;
        return _rawBuffers.Acquire(byteCount);
    }

    /// <summary>Returns a layer's previously acquired raw buffers to the reuse pool.</summary>
    private void ReleaseRawBuffers(IEnumerable<byte[]> buffers)
    {
        foreach (var buffer in buffers)
            _rawBuffers.Release(buffer);
    }

    /// <summary>Retained bytes currently held by the streaming buffer pool. For diagnostics.</summary>
    public long PoolRetainedBytes => _rawBuffers.RetainedBytes;

    /// <summary>Pool borrows that reused an existing buffer. For diagnostics.</summary>
    public long PoolHits => _rawBuffers.Hits;

    /// <summary>Pool borrows that allocated a fresh array. For diagnostics.</summary>
    public long PoolMisses => _rawBuffers.Misses;

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
    public override void InitializeWeights(IProgress<float>? progress = null, CancellationToken? cancellationToken = null)
    {
        var meta = Format.ModelFormatHelpers.LoadMetaForFile(GgufPath!);
        GgufMeta = meta;
        IsMoE = IsMoEGguf(meta.Tensors);

        // Populate TensorMeta for all blocks (file offsets, sizes, dtypes)
        foreach (var info in meta.Tensors)
        {
            var (target, block, rawField) = ResolveTarget(info.Name);

            // Gemma-style post-attention / post-FFN norms are 1D and carry no raw field,
            // so they are invisible to the raw-field scan below. BuildBlock creates their
            // NormLayers only when the tensor already exists, and a streaming load builds
            // the blocks before any layer data is read: without these the two norms are
            // silently dropped and Gemma-3 generates garbage. Pre-allocate the tensors so
            // BuildBlock wires the layers; the per-layer reload fills them in place (they
            // are kept resident across frees — see ReleaseLayerData).
            if (block != null)
            {
                if (info.Name.Contains("post_attention_norm", StringComparison.OrdinalIgnoreCase))
                    block.PostNorm1W ??= new Tensor<float>(Config.HiddenDim);
                else if (info.Name.Contains("post_ffw_norm", StringComparison.OrdinalIgnoreCase))
                    block.PostNorm2W ??= new Tensor<float>(Config.HiddenDim);
            }

            if (block != null && rawField != null)
                {
                    if (rawField == "RawWqkv")
                    {
                        // Fused QKV: register individual TensorMeta entries so the
                        // AttentionLayer constructor reads the correct dtype instead
                        // of defaulting to F32.
                        long qkvSize = Core.Quantization.QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
                        int partSize = (int)(qkvSize / 3);
                        long baseOffset = meta.DataOffset + info.Offset;
                        SetTensorMeta(block, "RawWq", baseOffset, partSize, info.Dtype);
                        SetTensorMeta(block, "RawWk", baseOffset + partSize, partSize, info.Dtype);
                        SetTensorMeta(block, "RawWv", baseOffset + partSize * 2, partSize, info.Dtype);
                    }
                    else if (rawField.EndsWith("ExpFused", StringComparison.Ordinal) && info.Shape.Length == 3)
                    {
                        // Fused MoE expert stack: the loader slices this 3D tensor into
                        // per-expert planes when the layer's bytes are read, but the
                        // FfnLayer is constructed from TensorMeta BEFORE that happens.
                        // Without these entries DtypeFromMeta("RawWgateExp_e") misses and
                        // every expert falls back to F32, so the quantized bytes are then
                        // rejected as the wrong size at load time.
                        //
                        // The plane size must come from the 2D sub-shape: the 3D byte
                        // count lays blocks along the last dim (the expert axis here),
                        // which over-counts unless numExperts is a multiple of the block
                        // size. See the matching note in GgufLoader.
                        int numExperts = info.Shape[2];
                        int planeSize = (int)Core.Quantization.QuantizationOps.GetRawTensorByteCount(
                            [info.Shape[0], info.Shape[1]], info.Dtype);
                        long baseOffset = meta.DataOffset + info.Offset;
                        string prefix = rawField switch
                        {
                            "RawWgateExpFused" => "RawWgateExp_",
                            "RawWupExpFused" => "RawWupExp_",
                            _ => "RawWdownExp_",
                        };
                        for (int e = 0; e < numExperts; e++)
                            SetTensorMeta(block, $"{prefix}{e}", baseOffset + (long)e * planeSize, planeSize, info.Dtype);
                    }
                    else
                    {
                        long rawSize = Core.Quantization.QuantizationOps.GetRawTensorByteCount(info.Shape, info.Dtype);
                        if (rawSize > 0)
                            SetTensorMeta(block, rawField, meta.DataOffset + info.Offset, (int)rawSize, info.Dtype);
                    }
                }
        }

        _pushedLayers.Clear();
        progress?.Report(1f);

        // Load global non-block tensors (embedding, final norm, lm_head).
        // These are not per-layer and must be present before any forward pass.
        Loader!.LoadGlobalTensors(this, cancellationToken);

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

        // Wait for an in-flight preload of THIS layer before inspecting its state.
        // LoadLayerWeights fills the block one tensor at a time, so checking Wq/RawWq
        // first can observe a half-filled block (attention bytes present, FFN still
        // missing) and skip the wait — pushing a partially loaded layer into the
        // TransformerBlock. Waiting on the task guarantees the whole block finished.
        lock (_preloadLock)
        {
            if (_preloadTask != null && _preloadLayerIndex == layerIndex)
            {
                _preloadTask.GetAwaiter().GetResult();
                _preloadTask = null;
                _preloadLayerIndex = -1;
            }
        }

        bool needsPush = false;

        // For fused QKV models (e.g. Phi-3), Wq is never set — only RawWq is.
        // Check both so the layer isn't reloaded every forward pass.
        if (Blocks[layerIndex].Wq == null && Blocks[layerIndex].RawWq == null)
        {
            Loader!.LoadLayerWeights(layerIndex, this);
            needsPush = true;
        }

        if (needsPush || !_pushedLayers.Contains(layerIndex))
        {
            BlockRefs?[layerIndex]?.SetWeights(Blocks[layerIndex]);
            _pushedLayers.Add(layerIndex);
        }

        // Learn the peak per-layer footprint so the pool budget covers the layers in the
        // resident window (free N−1 → acquire N+1 never crosses an eviction in steady state).
        long layerBytes = Blocks[layerIndex].AttachedRawBytes();
        if (layerBytes > _peakLayerBytes)
        {
            _peakLayerBytes = layerBytes;
            UpdatePoolBudget();
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

        // Snapshot the raw quantized buffers before they are cleared so they can be
        // recycled for the next layer that takes this one's place in the resident window.
        // Include layer-derived buffers (fused gate+up) registered on the block.
        var owned = Blocks[layerIndex].CaptureOwnedBuffers();
        var rawBuffers = Blocks[layerIndex].CaptureRawBuffers().ToArray();

        // Push empty weights into LinearLayers (clears raw data references)
        BlockRefs?[layerIndex]?.SetWeights(new BlockWeights());

        // Free float tensors in LinearLayers (disposes pre-allocated tensors)
        BlockRefs?[layerIndex]?.FreeFloatWeights();

        // Dispose float tensors and null all fields in BlockWeights; keeps TensorMeta
        Blocks[layerIndex].ReleaseLayerData();

        // Return the layer's raw buffers to the reuse pool (no-op reverse: a fresh load
        // acquires them again, so the hot rotation allocates nothing).
        ReleaseRawBuffers(rawBuffers);
        if (owned is { Count: > 0 }) ReleaseRawBuffers(owned);

        _pushedLayers.Remove(layerIndex);
    }

    /// <summary>
    /// Called after a forward pass once the block loop has finished. The loop has already
    /// kept a bounded resident window (layers near the end of the pass plus the preload for
    /// the next), so all layers must NOT be freed here: freeing everything forced every token
    /// to reload the whole model serially (measured at ~430 MiB of managed allocation and 3×
    /// wall time per decode step). Instead, warm the next pass by preloading layer 0, which
    /// the next pass's first block consumes. Layers inside <see cref="ResidentWindow"/> of the
    /// end stay loaded and their blocks skip the reload on the next pass. A small-window loop
    /// churns LOH-weight buffers each token, and with rare gen2 collections that churn would
    /// accumulate as live memory (measured +~30 MiB/token at window 1); pace a full collection
    /// between tokens so the footprint stays bounded — streaming is the low-memory mode and may
    /// trade speed for it, so a blocking collection here is the right trade.
    /// </summary>
    public void PrepareForNextForward()
    {
        MaybeCollectGen2();
        PreloadLayerAsync(0);
    }

    private void MaybeCollectGen2()
    {
        long now = GC.GetTotalAllocatedBytes();
        if (now - _lastGen2Check > Gen2CollectionBudget)
        {
            _lastGen2Check = now;
            GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: false);
        }
    }

    /// <summary>Allocate-budget between throttled gen2 collections (≈ one layer's worth of
    /// raw+derived buffers, so a window-1 token boundary triggers at most one).</summary>
    private const long Gen2CollectionBudget = 16L * 1024 * 1024;
    private long _lastGen2Check = -1;
}


