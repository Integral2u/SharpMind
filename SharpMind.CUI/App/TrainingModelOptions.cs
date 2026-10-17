using SharpMind.Core;
using SharpMind.Core.Quantization;
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
            // Pad the model's vocab (never the tokenizer's) up to the QAT target's
            // embedding block alignment, so a QAT run can't fail on the vocab axis.
            // The tokenizer keeps emitting ids below its own vocab, which stay valid
            // because the embedding table is only ever widened here.
            VocabSize = QatPaddedVocab(ParseQat(job.QuantAwareTraining), vocabSize, job.HiddenDim),
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

    /// <summary>
    /// Parses a stored QAT target name. Null/blank and "F32" mean QAT is off; an
    /// unrecognised name also maps to null (off) rather than throwing.
    /// </summary>
    public static QuantDType? ParseQat(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw)) return null;
        if (raw.Equals("F32", StringComparison.OrdinalIgnoreCase)) return null;
        return Enum.TryParse<QuantDType>(raw, ignoreCase: true, out var q) ? q : null;
    }

    /// <summary>
    /// Pre-flight check of a QAT target against the planned model dimensions, run
    /// before the tokenizer and model are built so a bad target fails fast with a
    /// clear message. Returns null when the target is compatible or off, otherwise
    /// a human-readable reason. Only <c>HiddenDim</c> can block a run — the vocab is
    /// padded to fit (see <see cref="QatPaddedVocab"/>).
    /// </summary>
    public static string? ValidateQat(string? quantAwareTraining, int hiddenDim)
    {
        var target = ParseQat(quantAwareTraining);
        if (target is null or QuantDType.F16) return null;

        if (target is QuantDType.Q8_0 or QuantDType.Q4_0)
            return hiddenDim % 32 == 0 ? null
                : $"QAT with {target} needs HiddenDim ({hiddenDim}) to be a multiple of 32. "
                  + "Use F16, pick a larger HiddenDim, or disable QAT.";

        if (!IsKQuant(target))
            return $"QAT target {target} is not supported for training. "
                 + "Supported targets: F16, Q8_0, Q4_0, Q2_K–Q8_K.";

        return hiddenDim % 128 == 0 ? null
            : $"QAT with {target} needs HiddenDim ({hiddenDim}) to be a multiple of 128. "
              + "Use F16, pick a larger HiddenDim, or disable QAT.";
    }

    /// <summary>
    /// Rounds a model vocab up to the smallest value that satisfies the QAT target's
    /// embedding block alignment: Q8_0/Q4_0 need <c>V % 32 == 0</c>; the K-quants need
    /// the flattened tied-head length <c>V*H</c> to be a multiple of 256, met by making
    /// V a multiple of <c>256 / gcd(256, H)</c>. Returns <paramref name="vocabSize"/>
    /// unchanged when QAT is off, F16, or the value is already aligned.
    /// </summary>
    public static int QatPaddedVocab(QuantDType? target, int vocabSize, int hiddenDim)
    {
        if (target is null or QuantDType.F16) return vocabSize;
        if (target is QuantDType.Q8_0 or QuantDType.Q4_0) return RoundUp(vocabSize, 32);
        if (!IsKQuant(target)) return vocabSize;

        int g = Gcd(256, hiddenDim);
        int step = g <= 0 ? 1 : 256 / g;
        return RoundUp(vocabSize, step);
    }

    private static bool IsKQuant(QuantDType? t) => t is
        QuantDType.Q2_K or QuantDType.Q2_K_S or
        QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L or
        QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M or
        QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M or
        QuantDType.Q6_K or QuantDType.Q6_K_S or QuantDType.Q8_K;

    private static int RoundUp(int value, int multiple)
        => multiple <= 1 ? value : (value + multiple - 1) / multiple * multiple;

    private static int Gcd(int a, int b)
    {
        while (b != 0) (a, b) = (b, a % b);
        return Math.Abs(a);
    }

    /// <summary>
    /// QAT target labels shown in the wizard, in menu order. The legacy block formats
    /// (Q8_0/Q4_0) sit next to the K-quants so a job that chose one keeps it — the old
    /// list only offered Q8/Q4 (K-quants) and silently rewrote a stored Q8_0/Q4_0 job to
    /// Q8_K/Q4_K on save.
    /// </summary>
    public static readonly string[] QatLabels =
        ["F32 (off)", "F16", "Q8_0", "Q4_0", "Q8_K", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K"];

    /// <summary>
    /// Index of the stored QAT target name in <see cref="QatLabels"/> (0 = off), or 0 for a
    /// null/blank/unknown value. Legacy names map to their own entry, never mutated to a
    /// different target.
    /// </summary>
    public static int QatIndexFor(string? qat)
    {
        if (string.IsNullOrEmpty(qat) || qat.Equals("F32", StringComparison.OrdinalIgnoreCase)) return 0;
        for (int i = 1; i < QatLabels.Length; i++)
            if (qat.Equals(QatLabels[i], StringComparison.OrdinalIgnoreCase)) return i;
        return 0;
    }

    /// <summary>The stored QAT target name for a <see cref="QatLabels"/> index; null = off.</summary>
    public static string? QatStoredName(int index)
        => index <= 0 || index >= QatLabels.Length ? null : QatLabels[index];
}