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
    /// Override for head dimension (per-head key/value size).
    /// If null, derived as HiddenDim / NumHeads.
    /// Qwen3 GGUF specifies explicit key_length (e.g. 128).
    /// </summary>
    public int? KeyLength { get; init; }

    /// <summary>Override for value head dimension. Falls back to <see cref="KeyLength"/> then derived.</summary>
    public int? ValueLength { get; init; }

    /// <summary>
    /// Explicit head_dim from GGUF ({arch}.head_dim).
    /// Some models (Gemma 2, DeepSeek V2) specify head_dim directly
    /// rather than deriving it from HiddenDim / NumHeads.
    /// When set and KeyLength is null, this takes priority for HeadDim.
    /// </summary>
    public int? HeadDimOverride { get; init; }

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
        if (NumExperts < TopKExperts)
            throw new InvalidOperationException(
                $"NumExperts ({NumExperts}) must be >= TopKExperts ({TopKExperts}).");
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
}
