namespace SharpMind.Model.Config;

/// <summary>
/// Immutable hyperparameter set for a transformer model.
/// Contains only dimensional and structural parameters — kernel selection
/// (attention variant, FFN kind, hardware tier) lives in <c>SharpMindConfig</c>.
///
/// Conventions:
///   HeadDim    = HiddenDim / NumHeads  (must divide evenly)
///   NumKvHeads = NumHeads for MHA, less for GQA, 1 for MQA
///   FfnDim     = HiddenDim * FfnMultiplier (typically 4 for dense, 2/3 * 4 for gated)
/// </summary>
public sealed record ModelConfig
{
    // Architecture

    /// <summary>
    /// GGUF architecture name (e.g. "qwen2", "llama", "bert", "gpt2").
    /// Set during GGUF loading; used by SharpMindConfig.ForModel to select
    /// the correct activation/gate/ffn/norm preset.
    /// </summary>
    public string? Architecture { get; init; }

    // Dimensions

    public PositionalEncoding PositionalEncoding { get; init; } = PositionalEncoding.RoPE;
    /// <summary>Vocabulary size — must match the tokenizer.</summary>
    public int VocabSize { get; init; }

    /// <summary>Token embedding and hidden state dimension.</summary>
    public int HiddenDim { get; init; }

    /// <summary>Number of transformer blocks.</summary>
    public int NumLayers { get; init; }

    /// <summary>Number of query attention heads.</summary>
    public int NumHeads { get; init; }

    /// <summary>
    /// Number of key/value heads.
    /// Equal to <see cref="NumHeads"/> for MHA.
    /// Less than <see cref="NumHeads"/> for GQA (e.g. LLaMA 3 8B uses 8).
    /// 1 for MQA.
    /// </summary>
    public int NumKvHeads { get; init; }

    /// <summary>FFN intermediate dimension. Typically HiddenDim × 4.</summary>
    public int FfnDim { get; init; }

    /// <summary>Maximum sequence length the model supports.</summary>
    public int MaxSeqLen { get; init; }

    /// <summary>
    /// True for models that mix sliding-window and full-attention layers
    /// (gemma-3/4 style). The windowed layers still mask attention to the last
    /// <see cref="SlidingWindowSize"/> tokens, but the full-attention layers
    /// attend the whole declared context, so the KV cache and RoPE tables must
    /// span <see cref="MaxSeqLen"/>, not just the window.
    /// </summary>
    public bool IsHybridSlidingWindow { get; init; }

    /// <summary>
    /// Effective context length for KV-cache allocation at inference.
    /// When a sliding window is declared, the cache is sized to the window
    /// (tokens beyond it are never attended to) — unless the model mixes
    /// windowed and full-attention layers (see <see cref="IsHybridSlidingWindow"/>),
    /// in which case the cache spans the full declared context. Falls back to
    /// <see cref="MaxSeqLen"/> for full-context models.
    /// </summary>
    public int EffectiveInferenceCacheLength =>
        SlidingWindowSize > 0 && !IsHybridSlidingWindow ? Math.Min(MaxSeqLen, SlidingWindowSize) : MaxSeqLen;

    /// <summary>
    /// Highest absolute position the model can represent (one past the last
    /// usable slot). This is the larger of the declared context and the cache
    /// capacity, matching the RoPE tables built by the attention layers.
    /// </summary>
    public int MaxPosition => Math.Max(MaxSeqLen, EffectiveInferenceCacheLength);

    /// <summary>
    /// Override for head dimension (per-head key/value size).
    /// If null, derived as HiddenDim / NumHeads.
    /// Qwen3 GGUF specifies explicit key_length (e.g. 128).
    /// </summary>
    public int? KeyLength { get; init; }

    /// <summary>Override for value head dimension. Falls back to <see cref="KeyLength"/> then derived.</summary>
    public int? ValueLength { get; init; }

    /// <summary>
    /// Sliding window attention size. When &gt; 0, each token attends only
    /// to the previous <see cref="SlidingWindowSize"/> tokens in the KV cache.
    /// 0 = full causal attention (default for most architectures).
    /// </summary>
    public int SlidingWindowSize { get; init; }

    /// <summary>
    /// Sliding-window pattern period (llama.cpp {arch}.attention.sliding_window_pattern).
    /// With llama.cpp's gemma-3 semantics (dense_first=false) a layer is
    /// full-attention when il % period == period - 1; every other layer is
    /// windowed. Null = legacy behaviour — when <see cref="SlidingWindowSize"/>
    /// is set, every layer is windowed (how SharpMind treated pre-gemma-3 files).
    /// </summary>
    public int? SlidingWindowPattern { get; init; }

    /// <summary>
    /// Explicit head_dim from GGUF ({arch}.head_dim).
    /// Some models (Gemma 2, DeepSeek V2) specify head_dim directly
    /// rather than deriving it from HiddenDim / NumHeads.
    /// When set and KeyLength is null, this takes priority for HeadDim.
    /// </summary>
    public int? HeadDimOverride { get; init; }

    /// <summary>
    /// Per-layer key/value head counts ({arch}.attention.head_count_kv array).
    /// When set, index i is the KV head count of block i and 0 marks a block
    /// without attention (e.g. LFM2 short-conv layers). When null, every block
    /// uses <see cref="NumKvHeads"/>.
    /// </summary>
    public int[]? LayerKvHeads { get; init; }

    /// <summary>
    /// Look-back length for the causal depthwise conv in no-attention layers
    /// (LFM2 short-conv). GGUF key: {arch}.shortconv.l_cache.
    /// </summary>
    public int ShortConvCacheLength { get; init; } = 3;

    // Qwen3.5 hybrid Gated DeltaNet

    /// <summary>Width of the causal depthwise conv in a Gated DeltaNet layer ({arch}.ssm.conv_kernel). 0 = not a hybrid SSM model.</summary>
    public int SsmConvKernel { get; init; }

    /// <summary>Value width of a Gated DeltaNet layer ({arch}.ssm.inner_size).</summary>
    public int SsmInnerSize { get; init; }

    /// <summary>Per-head key/value dimension of a Gated DeltaNet layer ({arch}.ssm.state_size).</summary>
    public int SsmStateSize { get; init; }

    /// <summary>Number of value heads of a Gated DeltaNet layer ({arch}.ssm.time_step_rank).</summary>
    public int SsmTimeStepRank { get; init; }

    /// <summary>Number of key/query heads of a Gated DeltaNet layer ({arch}.ssm.group_count).</summary>
    public int SsmGroupCount { get; init; }

    /// <summary>
    /// One full-attention layer every N blocks, with the other blocks being Gated
    /// DeltaNet linear-attention layers ({arch}.full_attention_interval). llama.cpp
    /// treats block i as linear when <c>(i + 1) % interval != 0</c>.
    /// </summary>
    public int? FullAttentionInterval { get; init; }

    /// <summary>True for the Qwen3.5 hybrid Gated DeltaNet + full-attention decoder.</summary>
    public bool IsQwen35 => Architecture?.StartsWith("qwen35", StringComparison.OrdinalIgnoreCase) == true;

    /// <summary>True when the model carries Qwen3.5 Gated DeltaNet metadata and can build recurrent layers.</summary>
    public bool HasDeltaNet =>
        IsQwen35 && SsmConvKernel > 1 && SsmInnerSize > 0 && SsmStateSize > 0 &&
        SsmTimeStepRank > 0 && SsmGroupCount > 0;

    /// <summary>True when the given block is a Gated DeltaNet (linear-attention) layer.</summary>
    public bool IsRecurrentLayer(int layerIndex) =>
        HasDeltaNet && ((layerIndex + 1) % (FullAttentionInterval ?? 4) != 0);

    /// <summary>True when the given block is a full softmax-attention layer.</summary>
    public bool IsFullAttentionLayer(int layerIndex) => !IsRecurrentLayer(layerIndex);

    /// <summary>Per-head value dimension of a Gated DeltaNet layer.</summary>
    public int SsmValueHeadDim => SsmTimeStepRank > 0 ? SsmInnerSize / SsmTimeStepRank : 0;

    /// <summary>Total key/query width of a Gated DeltaNet layer.</summary>
    public int SsmKeyDim => SsmGroupCount * SsmStateSize;

    /// <summary>Total value width of a Gated DeltaNet layer.</summary>
    public int SsmValueDim => SsmInnerSize;

    /// <summary>Fused Q/K/V projection width of a Gated DeltaNet layer.</summary>
    public int SsmConvDim => 2 * SsmKeyDim + SsmValueDim;

    /// <summary>
    /// True when full-attention layers fuse their output gate into the Q projection
    /// (Qwen3.5: attn_q emits [query | gate] per head, and the attention output is
    /// multiplied by sigmoid(gate) before the output projection).
    /// </summary>
    public bool GatedAttentionOutput { get; init; }

    /// <summary>Width of the Q projection for the given block (doubled by gated attention).</summary>
    public int AttentionQProjDimForLayer(int layerIndex) =>
        NumHeads * HeadDimForLayer(layerIndex) * (GatedAttentionOutput ? 2 : 1);

    // MoE

    /// <summary>Total number of experts. Ignored when FfnKind is not MoE.</summary>
    public int NumExperts { get; init; } = 8;

    /// <summary>Number of experts activated per token (top-k routing).</summary>
    public int TopKExperts { get; init; } = 2;

    /// <summary>
    /// Whether the top-k router weights are renormalised to sum to 1 before the
    /// expert outputs are accumulated.
    /// </summary>
    /// <remarks>
    /// This is architecture-dependent and is NOT equivalent to "missing key means
    /// normalise". llama.cpp hardcodes <c>norm_w = false</c> for qwen2moe
    /// (src/models/qwen2moe.cpp) and olmoe, so those architectures weight each token's
    /// experts by their raw softmax probabilities over <em>all</em> experts — the top-k
    /// weights sum to roughly 0.7-0.9, not 1. Only deepseek2 and exaone-moe read
    /// <c>expert_weights_norm</c> from metadata. Renormalising qwen2moe inflates the
    /// routed branch by ~1/0.8 relative to the correctly-gated shared expert and
    /// degrades generation to noise.
    /// Defaults to <see langword="false"/>, which matches the architectures verified
    /// against llama.cpp. Note that Mixtral has no longer any llama.cpp reference
    /// implementation upstream, so its behaviour cannot be verified this way.
    /// </remarks>
    public bool NormTopKProb { get; init; }

    /// <summary>
    /// Width of a single routed expert's FFN. GGUF key:
    /// {arch}.expert_feed_forward_length. 0 means "same as <see cref="FfnDim"/>".
    /// </summary>
    /// <remarks>
    /// In Qwen1.5-MoE this is 1408 while <see cref="FfnDim"/> is 5632 — the
    /// latter is the width of the *shared* expert, not a routed one. Sizing the
    /// routed experts from <see cref="FfnDim"/> would give every expert a
    /// weight guard that the file's expert planes cannot satisfy.
    /// </remarks>
    public int ExpertFfnDim { get; init; }

    /// <summary>
    /// Width of the always-on shared expert's FFN, or 0 when the model has no
    /// shared expert. GGUF key: {arch}.expert_shared_feed_forward_length
    /// (Qwen1.5-MoE: 5632).
    /// </summary>
    public int SharedExpertFfnDim { get; init; }

    /// <summary>Effective routed-expert FFN width, falling back to <see cref="FfnDim"/>.</summary>
    public int ResolvedExpertFfnDim => ExpertFfnDim > 0 ? ExpertFfnDim : FfnDim;

    /// <summary>
    /// Effective shared-expert FFN width, falling back to <see cref="FfnDim"/>. Used
    /// when a model ships *_shexp tensors without
    /// <c>expert_shared_feed_forward_length</c>, so the shared branch still gets a
    /// correctly sized matrix instead of a zero-width one.
    /// </summary>
    public int ResolvedSharedFfnDim => SharedExpertFfnDim > 0 ? SharedExpertFfnDim : FfnDim;

    /// <summary>True when the model carries an always-on shared expert alongside the routed ones.</summary>
    public bool HasSharedExpert => SharedExpertFfnDim > 0;

    // RoPE

    /// <summary>
    /// RoPE base frequency theta.
    /// LLaMA 2: 10_000. LLaMA 3: 500_000.
    /// </summary>
    public float RopeTheta { get; init; } = 10_000f;

    /// <summary>
    /// RoPE base frequency for sliding-window attention layers
    /// (llama.cpp {arch}.rope.freq_base_swa). Gemma-3 uses 10_000 for
    /// sliding-window layers while full-attention layers keep
    /// <see cref="RopeTheta"/> (1_000_000). llama.cpp defaults the SWA base
    /// to 10_000 when the GGUF omits the key. When null, sliding-window
    /// layers use <see cref="RopeTheta"/>.
    /// </summary>
    public float? RopeThetaSwa { get; init; }

    /// <summary>
    /// RoPE dimension count. If non-null, only the first N dimensions
    /// of each head receive rotary encoding (partial RoPE).
    /// </summary>
    public int? RopeDim { get; init; }

    /// <summary>
    /// RoPE scaling type from GGUF ({arch}.rope.scaling.type).
    /// "linear", "dynamic", "yarn", etc. Null means no scaling.
    /// </summary>
    public string? RopeScalingType { get; init; }

    /// <summary>RoPE scaling factor (e.g. 2.0 for 2x context extension).</summary>
    public float? RopeScalingFactor { get; init; }

    /// <summary>Original max sequence length before RoPE scaling was applied.</summary>
    public int? RopeOriginalContextLength { get; init; }

    /// <summary>NTK-by-parts low frequency factor (llama3 scaling).</summary>
    public float? RopeLowFreqFactor { get; init; }

    /// <summary>NTK-by-parts high frequency factor (llama3 scaling).</summary>
    public float? RopeHighFreqFactor { get; init; }

    /// <summary>Pre-computed RoPE frequencies from GGUF (rope_freqs.weight), if present.</summary>
    public float[]? PrecomputedRopeFreqs { get; init; }

    // Gemma 3n / Gemma 4 family

    /// <summary>
    /// Per-layer embedding dimension (gemma4 <c>{arch}.embedding_length_per_layer_input</c>).
    /// Each block consumes a distinct slice of a per-layer token-embedding table,
    /// combined with a projection of the hidden state. 0 = architecture has no
    /// per-layer embeddings.
    /// </summary>
    public int PerLayerEmbdDim { get; init; }

    /// <summary>True when the model carries per-layer input embeddings (gemma4).</summary>
    public bool HasPerLayerEmbeddings => PerLayerEmbdDim > 0;

    /// <summary>
    /// Total width of the per-layer token-embedding table's rows
    /// (<see cref="PerLayerEmbdDim"/> × <see cref="NumLayers"/>). One row of
    /// <c>per_layer_token_embd.weight</c> holds every layer's slice back to
    /// back, layer-major.
    /// </summary>
    public int PerLayerTotalDim => PerLayerEmbdDim * NumLayers;

    /// <summary>
    /// Per-head key dimension for sliding-window layers
    /// (gemma4 <c>{arch}.attention.key_length_swa</c>). Gemma 4 uses a smaller
    /// head dim (256) on windowed layers than on full-attention layers (512).
    /// Null falls back to <see cref="KeyLength"/>.
    /// </summary>
    public int? KeyLengthSwa { get; init; }

    /// <summary>Per-head value dim for sliding-window layers. Falls back to <see cref="KeyLengthSwa"/>.</summary>
    public int? ValueLengthSwa { get; init; }

    /// <summary>
    /// RoPE dimension count for sliding-window layers
    /// (gemma4 <c>{arch}.rope.dimension_count_swa</c>). Null falls back to
    /// <see cref="RopeDim"/> then the layer's head dim.
    /// </summary>
    public int? RopeDimSwa { get; init; }

    /// <summary>
    /// Number of trailing layers that reuse the KV cache of earlier layers
    /// (gemma4 <c>{arch}.attention.shared_kv_layers</c>). 0 = every layer owns
    /// its own K/V.
    /// </summary>
    public int SharedKvLayers { get; init; }

    /// <summary>True when trailing layers share the KV cache of earlier layers.</summary>
    public bool HasSharedKv => SharedKvLayers > 0;

    /// <summary>Index of the first layer that shares the KV cache; layers at or beyond reuse earlier ones.</summary>
    public int FirstKvSharedLayer => NumLayers - SharedKvLayers;

    /// <summary>
    /// Override for the attention logit scale. Null derives
    /// <c>1 / sqrt(headDim)</c> from the layer's head dim; gemma4 sets it to
    /// 1.0 (no pre-attention scaling, llama.cpp <c>f_attention_scale</c>).
    /// </summary>
    public float? AttentionScale { get; init; }

    /// <summary>
    /// gemma4 applies an unweighted RMSNorm (llama.cpp
    /// <c>ggml_rms_norm(v, eps)</c>) to the value vectors before they are
    /// cached. Unlike the Q/K norms this has no learned weight.
    /// </summary>
    public bool RmsNormValue { get; init; }

    /// <summary>
    /// Per-pair RoPE frequency factors (llama.cpp frequency factors, gemma4
    /// <c>rope_freqs.weight</c>). The effective rotation angle for pair i is
    /// <c>pos * theta^(-2i/ropeDim) / RopeFreqFactors[i]</c>; a factor of 1
    /// leaves the pair unchanged, a very large factor effectively disables the
    /// rotation (gemma4 uses this for proportional RoPE on full-attention
    /// layers). Null disables the factor entirely.
    /// </summary>
    public float[]? RopeFreqFactors { get; init; }

    /// <summary>
    /// Final logit soft-capping value (gemma2/3/4
    /// <c>{arch}.final_logit_softcapping</c>): logits are passed through
    /// <c>tanh(x / cap) * cap</c> before sampling. 0 = no soft-capping.
    /// </summary>
    public float FinalLogitSoftcap { get; init; }

    /// <summary>True when the model soft-caps its final logits (Gemma 2/3/4).</summary>
    public bool HasFinalLogitSoftcap => FinalLogitSoftcap > 0;

    /// <summary>Whether the LM head weight is tied to the embedding weight.</summary>
    public bool? TieWordEmbeddings { get; init; }

    // Multimodal encoders (vision / audio)

    /// <summary>
    /// Vision patch size (e.g. 16). A positive value enables the vision
    /// encoder, which splits an <see cref="VisionImageSize"/>² input image into
    /// patches of this size and projects each flattened patch to HiddenDim.
    /// </summary>
    public int? VisionPatchSize { get; init; }

    /// <summary>Square image side length in pixels. Ignored when <see cref="VisionPatchSize"/> is null.</summary>
    public int VisionImageSize { get; init; } = 224;

    /// <summary>Number of image channels (3 = RGB). Ignored when <see cref="VisionPatchSize"/> is null.</summary>
    public int VisionChannels { get; init; } = 3;

    /// <summary>
    /// Mel-spectrogram bin count (e.g. 80). A positive value enables the audio
    /// encoder, which projects each mel frame to HiddenDim.
    /// </summary>
    public int? AudioMelBins { get; init; }

    /// <summary>Maximum number of audio frames the encoder supports. Ignored when <see cref="AudioMelBins"/> is null.</summary>
    public int AudioMaxFrames { get; init; } = 1_024;

    /// <summary>True when the vision encoder is enabled.</summary>
    public bool HasVision => VisionPatchSize is > 0;

    /// <summary>Number of image patches for the configured image/patch sizes.</summary>
    public int VisionNumPatches => !HasVision ? 0
        : VisionImageSize / VisionPatchSize!.Value * (VisionImageSize / VisionPatchSize!.Value);

    /// <summary>Flattened size of a single image patch (channels × patch²).</summary>
    public int VisionPatchDim => !HasVision ? 0
        : VisionChannels * VisionPatchSize!.Value * VisionPatchSize!.Value;

    /// <summary>True when the audio encoder is enabled.</summary>
    public bool HasAudio => AudioMelBins is > 0;

    // Encoder pooling / token types

    /// <summary>
    /// Encoder sequence-pooling strategy used to collapse the final-normed
    /// hidden states into one sentence embedding. Read from GGUF
    /// ({arch}.pooling_type). <see cref="Config.PoolingType.None"/> for models
    /// that declare none (all decoders).
    /// </summary>
    public PoolingType Pooling { get; init; } = PoolingType.None;

    /// <summary>
    /// Number of token-type (segment) embedding rows
    /// (tokenizer.ggml.token_type_count). 0 when the model has no token-type
    /// table. BERT-family models use this for sentence-pair segments; a
    /// single-sequence encode uses type 0 for every token.
    /// </summary>
    public int NumTokenTypes { get; init; }

    /// <summary>True when the model carries a token-type embedding table.</summary>
    public bool HasTokenTypes => NumTokenTypes > 0;

    /// <summary>
    /// BERT-style post-norm residual placement: the block applies no pre-norm and
    /// instead normalises after each residual add
    /// (<c>x = norm(x + sublayer(x))</c>), with the FFN reading the
    /// post-attention-normed state. False for the standard pre-norm blocks.
    /// </summary>
    public bool PostNormResidual { get; init; }

    // Normalisation

    /// <summary>Epsilon for RMS / LayerNorm. Qwen2 uses 1e-6; LLaMA 2 uses 1e-5.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>
    /// Norm type override from GGUF ({arch}.norm_type).
    /// 0 = LayerNorm, 1 = RMSNorm.
    /// When non-null, overrides the architecture-based default in SharpMindConfig.
    /// </summary>
    public int? NormTypeOverride { get; init; }

    // Derived

    /// <summary>Dimension per query head. Priority: KeyLength > HeadDimOverride > HiddenDim / NumHeads.</summary>
    public int HeadDim => KeyLength ?? HeadDimOverride ?? HiddenDim / NumHeads;

    /// <summary>Dimension per value head. Falls back to HeadDim.</summary>
    public int ValueDim => ValueLength ?? HeadDim;

    /// <summary>
    /// Per-head key dimension for the given block. Sliding-window layers of
    /// gemma4 use <see cref="KeyLengthSwa"/> (256) while full-attention layers
    /// use <see cref="HeadDim"/> (512). Null SWA override falls back to <see cref="HeadDim"/>.
    /// </summary>
    public int HeadDimForLayer(int layerIndex) =>
        IsSwaLayer(layerIndex) && KeyLengthSwa is { } k ? k : HeadDim;

    /// <summary>Per-head value dimension for the given block (see <see cref="HeadDimForLayer"/>).</summary>
    public int ValueDimForLayer(int layerIndex) =>
        IsSwaLayer(layerIndex) && ValueLengthSwa is { } v ? v
        : IsSwaLayer(layerIndex) && KeyLengthSwa is { } k ? k
        : ValueDim;

    /// <summary>
    /// Rotary span for the given block: the SWA override on windowed layers,
    /// else <see cref="RopeDim"/>, else the layer's head dim (full rotation).
    /// </summary>
    public int RopeDimForLayer(int layerIndex) =>
        IsSwaLayer(layerIndex) && RopeDimSwa is { } r ? r
        : RopeDim ?? HeadDimForLayer(layerIndex);

    /// <summary>
    /// True when the given block computes its own K/V. When trailing layers
    /// share the KV cache (<see cref="HasSharedKv"/>), those layers return false
    /// and reuse an earlier layer's keys and values during attention.
    /// </summary>
    public bool HasOwnKv(int layerIndex) => !HasSharedKv || layerIndex < FirstKvSharedLayer;

    /// <summary>
    /// Layer whose KV cache a KV-sharing layer reads (gemma4). Shared layers
    /// reuse the last non-shared layer <em>of the same attention type</em>
    /// (sliding-window vs full), reproducing vLLM's gemma4 mapping: for the
    /// E4B checkpoint the shared SWA layers read layer 22 and the shared full
    /// layers read layer 23. Layers that own their KV return themselves.
    /// </summary>
    public int KvSourceLayer(int layerIndex)
    {
        if (HasOwnKv(layerIndex)) return layerIndex;
        bool swa = IsSwaLayer(layerIndex);
        for (int j = FirstKvSharedLayer - 1; j >= 0; j--)
            if (HasOwnKv(j) && IsSwaLayer(j) == swa) return j;
        return FirstKvSharedLayer - 1;
    }

    /// <summary>Number of query heads each KV head serves (GQA group size).</summary>
    public int KvGroupSize => NumHeads / NumKvHeads;

    /// <summary>True when the given block is a no-attention (short-conv) layer.</summary>
    public bool IsShortConvLayer(int layerIndex) =>
        LayerKvHeads != null &&
        (uint)layerIndex < (uint)LayerKvHeads.Length &&
        LayerKvHeads[layerIndex] == 0;

    /// <summary>
    /// True when the given block uses sliding-window attention (window mask
    /// plus the SWA RoPE base). Without a declared pattern every layer of a
    /// windowed model is treated as SWA (legacy behaviour).
    /// </summary>
    public bool IsSwaLayer(int layerIndex)
    {
        if (SlidingWindowSize <= 0) return false;
        int period = SlidingWindowPattern ?? 0;
        if (period <= 0) return true;
        // llama.cpp set_swa_pattern(n, dense_first: false):
        // SWA unless il % n == n - 1 (Gemma-3 full-attention layers).
        return (uint)layerIndex % (uint)period != (uint)period - 1;
    }

    /// <summary>
    /// RoPE base frequency for the given block. Sliding-window layers of
    /// models with <see cref="RopeThetaSwa"/> (e.g. Gemma-3) use the SWA base
    /// (10_000); full-attention layers use <see cref="RopeTheta"/> (1_000_000).
    /// </summary>
    public float RopeThetaForLayer(int layerIndex) =>
        IsSwaLayer(layerIndex) && RopeThetaSwa is { } swa ? swa : RopeTheta;

    /// <summary>
    /// Attention window for the given block: <see cref="SlidingWindowSize"/>
    /// for sliding-window layers, 0 for full-attention layers.
    /// </summary>
    public int WindowSizeForLayer(int layerIndex) =>
        IsSwaLayer(layerIndex) ? SlidingWindowSize : 0;


    // Validation

    public void Validate()
    {
        if (VocabSize <= 0) throw new InvalidOperationException("VocabSize must be > 0.");
        if (HiddenDim <= 0) throw new InvalidOperationException("HiddenDim must be > 0.");
        if (NumLayers <= 0) throw new InvalidOperationException("NumLayers must be > 0.");
        if (NumHeads <= 0) throw new InvalidOperationException("NumHeads must be > 0.");
        if (NumKvHeads <= 0) throw new InvalidOperationException("NumKvHeads must be > 0.");
        if (FfnDim <= 0) throw new InvalidOperationException("FfnDim must be > 0.");
        if (MaxSeqLen <= 0) throw new InvalidOperationException("MaxSeqLen must be > 0.");
        if (HiddenDim % NumHeads != 0)
            throw new InvalidOperationException(
                $"HiddenDim ({HiddenDim}) must be divisible by NumHeads ({NumHeads}).");
        if (NumHeads % NumKvHeads != 0)
            throw new InvalidOperationException(
                $"NumHeads ({NumHeads}) must be divisible by NumKvHeads ({NumKvHeads}).");
        if (LayerKvHeads != null && LayerKvHeads.Length != NumLayers)
            throw new InvalidOperationException(
                $"LayerKvHeads must have one entry per layer ({LayerKvHeads.Length} != {NumLayers}).");
        if (LayerKvHeads != null && LayerKvHeads.Any(v => v < 0))
            throw new InvalidOperationException(
                "LayerKvHeads entries must be >= 0 (0 marks a no-attention layer).");
        if (ShortConvCacheLength < 1)
            throw new InvalidOperationException($"ShortConvCacheLength must be >= 1 (was {ShortConvCacheLength}).");
        if (HasDeltaNet)
        {
            if (SsmTimeStepRank <= 0 || SsmInnerSize % SsmTimeStepRank != 0)
                throw new InvalidOperationException(
                    $"SsmInnerSize ({SsmInnerSize}) must be divisible by SsmTimeStepRank ({SsmTimeStepRank}).");
            if (SsmGroupCount <= 0 || SsmTimeStepRank % SsmGroupCount != 0)
                throw new InvalidOperationException(
                    $"SsmTimeStepRank ({SsmTimeStepRank}) must be divisible by SsmGroupCount ({SsmGroupCount}).");
            if (FullAttentionInterval is int fai && fai <= 1)
                throw new InvalidOperationException($"FullAttentionInterval must be > 1 (was {fai}).");
        }
        if (SlidingWindowPattern is int period && period <= 0)
            throw new InvalidOperationException($"SlidingWindowPattern must be > 0 (was {period}).");
        if (SlidingWindowPattern is not null && SlidingWindowSize <= 0)
            throw new InvalidOperationException("SlidingWindowPattern requires a SlidingWindowSize > 0.");
        if (IsHybridSlidingWindow && SlidingWindowSize <= 0)
            throw new InvalidOperationException("IsHybridSlidingWindow requires a SlidingWindowSize > 0.");
        if (IsHybridSlidingWindow && (SlidingWindowPattern is null || SlidingWindowPattern <= 0))
            throw new InvalidOperationException("IsHybridSlidingWindow requires a SlidingWindowPattern > 0.");
        if (NumExperts < TopKExperts)
            throw new InvalidOperationException(
                $"NumExperts ({NumExperts}) must be >= TopKExperts ({TopKExperts}).");
        if (ExpertFfnDim is < 0)
            throw new InvalidOperationException($"ExpertFfnDim must be >= 0 (was {ExpertFfnDim}); 0 means 'same as FfnDim'.");
        if (SharedExpertFfnDim < 0)
            throw new InvalidOperationException(
                $"SharedExpertFfnDim must be >= 0 (was {SharedExpertFfnDim}); 0 means 'no shared expert'.");
        if (HasVision)
        {
            if (VisionImageSize <= 0)
                throw new InvalidOperationException("VisionImageSize must be > 0 when the vision encoder is enabled.");
            if (VisionImageSize % VisionPatchSize!.Value != 0)
                throw new InvalidOperationException(
                    $"VisionImageSize ({VisionImageSize}) must be divisible by VisionPatchSize ({VisionPatchSize}).");
            if (VisionChannels <= 0)
                throw new InvalidOperationException("VisionChannels must be > 0 when the vision encoder is enabled.");
        }
        if (HasAudio && AudioMaxFrames <= 0)
            throw new InvalidOperationException("AudioMaxFrames must be > 0 when the audio encoder is enabled.");
        if (HasAudio && AudioMelBins!.Value <= 0)
            throw new InvalidOperationException("AudioMelBins must be > 0 when the audio encoder is enabled.");
    }

    // Presets

    /// <summary>GPT-2 small (117M parameters).</summary>
    public static ModelConfig Gpt2Small => new()
    {
        VocabSize = 50_257,
        HiddenDim = 768,
        NumLayers = 12,
        NumHeads = 12,
        NumKvHeads = 12,
        FfnDim = 3_072,
        MaxSeqLen = 1_024,
    };

    /// <summary>LLaMA 3 8B approximate config.</summary>
    public static ModelConfig Llama3_8B => new()
    {
        VocabSize = 128_256,
        HiddenDim = 4_096,
        NumLayers = 32,
        NumHeads = 32,
        NumKvHeads = 8,
        FfnDim = 14_336,
        MaxSeqLen = 8_192,
        RopeTheta = 500_000f,
    };

    /// <summary>Tiny debug config — fast to construct and run in tests.</summary>
    public static ModelConfig Tiny => new()
    {
        VocabSize = 512,
        HiddenDim = 64,
        NumLayers = 2,
        NumHeads = 4,
        NumKvHeads = 4,
        FfnDim = 256,
        MaxSeqLen = 128,
    };

    /// <summary>Minimal learnable config for training validation.</summary>
    public static ModelConfig Learnable => new()
    {
        VocabSize = 64,
        HiddenDim = 32,
        NumLayers = 1,
        NumHeads = 4,
        NumKvHeads = 4,
        FfnDim = 64,
        MaxSeqLen = 16,
    };

    /// <summary>
    /// Computes a safe KV cache length that fits in available memory,
    /// applying the user-specified cap if provided.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="userMaxCacheLen">User-specified cap (from CLI --max-cache-len). Null means no explicit cap.</param>
    /// <param name="headDim">Per-head dimension for KV tensors. Uses config.HeadDim when null.</param>
    /// <returns>The maximum cache length to use for KV cache allocation.</returns>
    public static int ComputeMaxCacheLength(ModelConfig config, int? userMaxCacheLen = null, int? headDim = null)
    {
        int effectiveLen = config.EffectiveInferenceCacheLength;
        int hd = headDim ?? config.HeadDim;

        // Bytes per token position in one layer: keys + values, each float32
        long bytesPerPosition = (long)config.NumKvHeads * hd * sizeof(float) * 2;
        long totalBytes = bytesPerPosition * effectiveLen * config.NumLayers;

        // Use at most 40% of available memory for KV cache (leaves room for
        // model weights, workspace, OS overhead, and other allocations).
        long availableBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        long budget = (long)(availableBytes * 0.40);

        int maxLen = effectiveLen;
        if (totalBytes > budget && bytesPerPosition > 0)
            maxLen = (int)Math.Max(1, budget / (bytesPerPosition * config.NumLayers));

        // Clamp to the user-specified cap (if any)
        if (userMaxCacheLen is int userMax && userMax > 0)
            maxLen = Math.Min(maxLen, userMax);

        return Math.Min(maxLen, effectiveLen);
    }
}
