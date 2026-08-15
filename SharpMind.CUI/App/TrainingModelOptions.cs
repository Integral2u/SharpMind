using SharpMind.Core;
using SharpMind.Model.Config;

namespace SharpMind.CUI.App;

/// <summary>
/// One selectable architecture preset for the training wizard. Each preset
/// carrie the kernel-option family (activation/gate/ffn/norm/arch/attention)
/// plus sensible dimension defaults so a one-click choice produces a coherent,
/// trainable model. <see cref="ArchitecturePresetKey"/> maps to the GGUF
/// architecture names understood by <see cref="SharpMindConfig.ForModel"/> and
/// is stamped onto exported models so loading reproduces the same family.
/// </summary>
public sealed record TrainingPreset(
    string ArchitecturePresetKey,
    string DisplayName,
    ActivationKind Activation,
    GateKind Gate,
    FfnKind Ffn,
    AttentionKind? Attention,
    NormKind Norm,
    ArchKind Arch,
    PositionalEncoding PositionalEncoding,
    int HiddenDim,
    int NumLayers,
    int NumHeads,
    int NumKvHeads,
    int FfnDim,
    int MaxSeqLen,
    int NumExperts = 8,
    int TopKExperts = 2);

/// <summary>
/// Resolves the architecture/optimizer options on a <see cref="TrainJobSettings"/>
/// into the <see cref="SharpMindConfig"/> and <see cref="ModelConfig"/> that
/// <see cref="TrainRunner"/> threads into model construction and training.
/// </summary>
public static class TrainingModelOptions
{
    public static TrainingPreset[] Presets { get; } =
    [
        new(
            "gpt2", "GPT-2 (dense, LayerNorm)",
            ActivationKind.GELU, GateKind.None, FfnKind.Dense, AttentionKind.MHA,
            NormKind.LayerNorm, ArchKind.Decoder, PositionalEncoding.RoPE,
            128, 4, 8, 8, 512, 256),
        new(
            "llama", "LLaMA (gated, RMSNorm)",
            ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.Gated, AttentionKind.GQA,
            NormKind.RMSNorm, ArchKind.Decoder, PositionalEncoding.RoPE,
            128, 4, 8, 2, 512, 256),
        new(
            "bert", "BERT (dense encoder, LayerNorm)",
            ActivationKind.GELU, GateKind.None, FfnKind.Dense, AttentionKind.MHA,
            NormKind.LayerNorm, ArchKind.Encoder, PositionalEncoding.NoPE,
            128, 4, 8, 8, 512, 256),
        new(
            "qwen3", "Qwen (gated, MQA, RMSNorm)",
            ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.Gated, AttentionKind.MQA,
            NormKind.RMSNorm, ArchKind.Decoder, PositionalEncoding.RoPE,
            128, 4, 8, 1, 512, 256),
        new(
            "mixtral", "Mixtral (MoE, RMSNorm)",
            ActivationKind.SiLU, GateKind.SwiGLU, FfnKind.MoE, AttentionKind.GQA,
            NormKind.RMSNorm, ArchKind.Decoder, PositionalEncoding.RoPE,
            128, 4, 8, 2, 512, 256,
            8, 2),
    ];

    /// <summary>Applies <paramref name="preset"/> onto <paramref name="job"/>, filling both dimensions and kernel options.</summary>
    public static void Apply(TrainJobSettings job, TrainingPreset preset)
    {
        job.ArchitecturePreset = preset.ArchitecturePresetKey;
        job.Activation = preset.Activation.ToString();
        job.Gate = preset.Gate.ToString();
        job.Ffn = preset.Ffn.ToString();
        job.Attention = preset.Attention?.ToString();
        job.Norm = preset.Norm.ToString();
        job.Arch = preset.Arch.ToString();
        job.PositionalEncoding = preset.PositionalEncoding.ToString();
        job.HiddenDim = preset.HiddenDim;
        job.NumLayers = preset.NumLayers;
        job.NumHeads = preset.NumHeads;
        job.NumKvHeads = preset.NumKvHeads;
        job.FfnDim = preset.FfnDim;
        job.MaxSeqLen = preset.MaxSeqLen;
        job.NumExperts = preset.NumExperts;
        job.TopKExperts = preset.TopKExperts;
    }

    /// <summary>
    /// Builds the <see cref="SharpMindConfig"/> for a job's architecture options.
    /// Without a preset it resolves to GPT-2-style defaults (the historic
    /// behaviour), so saved jobs without the new fields train exactly as before.
    /// </summary>
    public static SharpMindConfig ResolveSharpConfig(TrainJobSettings job)
    {
        var cfg = job.ArchitecturePreset is not null
            ? SharpMindConfig.ForModel(
                numHeads: job.NumHeads > 0 ? job.NumHeads : 8,
                numKvHeads: job.NumKvHeads > 0 ? job.NumKvHeads : job.NumHeads > 0 ? job.NumHeads : 8,
                architecture: job.ArchitecturePreset,
                hw: HardwareTier.Auto)
            : SharpMindConfig.Gpt with { Hardware = HardwareTier.Auto };

        return cfg with
        {
            Activation = ParseEnum(job.Activation, cfg.Activation),
            Gate = ParseEnum(job.Gate, cfg.Gate),
            Ffn = ParseEnum(job.Ffn, cfg.Ffn),
            Norm = ParseEnum(job.Norm, cfg.Norm),
            Arch = ParseEnum(job.Arch, cfg.Arch),
            Attention = job.Attention is not null && Enum.TryParse<AttentionKind>(job.Attention, true, out var a)
                ? a
                : cfg.Attention,
            Hardware = HardwareTier.Auto,
        };
    }

    /// <summary>Builds the option-derived fields of <see cref="ModelConfig"/> for a job.</summary>
    public static ModelConfig ResolveModelConfig(TrainJobSettings job, int vocabSize)
    {
        return new ModelConfig
        {
            VocabSize = vocabSize,
            HiddenDim = job.HiddenDim,
            NumLayers = job.NumLayers,
            NumHeads = job.NumHeads,
            NumKvHeads = job.NumKvHeads,
            FfnDim = job.FfnDim,
            MaxSeqLen = job.MaxSeqLen,
            NormEps = job.NormEps,
            NumExperts = job.NumExperts,
            TopKExperts = job.TopKExperts,
            PositionalEncoding = job.PositionalEncoding is not null
                && Enum.TryParse<PositionalEncoding>(job.PositionalEncoding, true, out var pe)
                ? pe
                : PositionalEncoding.RoPE,
        };
    }

    /// <summary>True when the job selects SGD rather than AdamW.</summary>
    public static bool UsesSgd(TrainJobSettings job)
        => !string.IsNullOrWhiteSpace(job.Optimizer)
           && job.Optimizer.Equals("SGD", StringComparison.OrdinalIgnoreCase);

    private static T ParseEnum<T>(string? name, T fallback) where T : struct, Enum
        => name is not null && Enum.TryParse<T>(name, true, out var value) ? value : fallback;
}