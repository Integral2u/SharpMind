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
    /// Effective context length for KV-cache allocation at inference.
    /// When a sliding window is declared, the cache is sized to the window
    /// (tokens beyond it are never attended to). Falls back to
    /// <see cref="MaxSeqLen"/> for full-context models.
    /// </summary>
    public int EffectiveInferenceCacheLength =>
        SlidingWindowSize > 0 ? Math.Min(MaxSeqLen, SlidingWindowSize) : MaxSeqLen;

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

    // MoE

    /// <summary>Total number of experts. Ignored when FfnKind is not MoE.</summary>
    public int NumExperts { get; init; } = 8;

    /// <summary>Number of experts activated per token (top-k routing).</summary>
    public int TopKExperts { get; init; } = 2;

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
        if (SlidingWindowPattern is int period && period <= 0)
            throw new InvalidOperationException($"SlidingWindowPattern must be > 0 (was {period}).");
        if (SlidingWindowPattern is not null && SlidingWindowSize <= 0)
            throw new InvalidOperationException("SlidingWindowPattern requires a SlidingWindowSize > 0.");
        if (NumExperts < TopKExperts)
            throw new InvalidOperationException(
                $"NumExperts ({NumExperts}) must be >= TopKExperts ({TopKExperts}).");
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
