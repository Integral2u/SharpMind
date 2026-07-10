using System.Runtime.Intrinsics.X86;

namespace SharpMind;

public sealed record SharpMindConfig
{
    // JigSaw Pointers (Abstract method names)
    public const string PtrPointWise = "ApplyPointwise";
    public const string PtrGate = "ApplyGate";
    public const string PtrRMSNorm = "ApplyRMSNormRow";
    public const string PtrSoftmax = "ApplySoftmaxRow";

    public const string PtrAttention = "ScaledDotProduct";
    public const string PtrFfn = "ApplyFfn";
    public const string PtrNorm = "ApplyRow";
    public const string PtrArch = "Forward";

    //  Activation Keys
    public const string KeyPointWise = "pointwise";
    public const string KeyGate = "gate";
    public const string KeyRMSNorm = "rmsnorm";
    public const string KeySoftmax = "softmax";

    // Model Layer Keys
    public const string KeyAttention = "attention";
    public const string KeyAttentionQ8 = "attention_q8";
    public const string KeyAttentionQ4 = "attention_q4";
    public const string KeyFfn = "ffn";
    public const string KeyNorm = "norm";
    public const string KeyArch = "arch";
    // Training Keys
    public const string KeyAdamW = "adamw";
    public const string KeyGradNorm = "gradnorm";
    // Norm Row Keys
    public const string KeyLayerNormRow = "layernormrow";

    // Activation Values
    public const string ValReLU = "relu";
    public const string ValReLUAvx2 = "reluavx2";
    public const string ValReLUFma = "relufma";
    public const string ValGELU = "gelu";
    public const string ValGELUAvx2 = "geluavx2";
    public const string ValGELUFma = "gelufma";
    public const string ValSiLU = "silu";
    public const string ValSiLUAvx2 = "siluavx2";
    public const string ValSiLUFma = "silufma";
    public const string ValSwiGLU = "swiglu";
    public const string ValSwiGLUAvx2 = "swigluavx2";
    public const string ValSwiGLUFma = "swiglufma";
    public const string ValGeGLU = "geglu";
    public const string ValGeGLUAvx2 = "gegluavx2";
    public const string ValGeGLUFma = "geglufma";
    public const string ValNone = "none";
    public const string ValNoneAvx2 = "noneavx2";
    public const string ValNoneFma = "nonefma";

    // Hardware Values
    public const string ValScalar = "scalar";
    public const string ValAvx2 = "avx2";
    public const string ValFma = "fma";

    // Attention Values
    public const string ValMhaAvx2 = "mhaavx2";
    public const string ValMhaFma = "mhafma";
    public const string ValMhaScalar = "mhascalar";
    public const string ValMhaFlashAvx2 = "mhaflashavx2";
    public const string ValMhaFlashFma = "mhaflashfma";
    public const string ValMhaFlashScalar = "mhaflashscalar";
    public const string ValGqaAvx2 = "gqaavx2";
    public const string ValGqaFma = "gqafma";
    public const string ValGqaScalar = "gqascalar";
    public const string ValGqaFlashAvx2 = "gqaflashavx2";
    public const string ValGqaFlashFma = "gqaflashfma";
    public const string ValGqaFlashScalar = "gqaflashscalar";
    public const string ValMqaAvx2 = "mqaavx2";
    public const string ValMqaFma = "mqafma";
    public const string ValMqaScalar = "mqascalar";
    public const string ValMqaFlashAvx2 = "mqaflashavx2";
    public const string ValMqaFlashFma = "mqaflashfma";
    public const string ValMqaFlashScalar = "mqaflashscalar";

    // Q8_0 Attention Values
    public const string ValMhaFlashQ8_0Avx2 = "mhaflashq8_0avx2";
    public const string ValMhaFlashQ8_0Fma = "mhaflashq8_0fma";
    public const string ValMhaFlashQ8_0Scalar = "mhaflashq8_0scalar";
    public const string ValGqaFlashQ8_0Avx2 = "gqaflashq8_0avx2";
    public const string ValGqaFlashQ8_0Fma = "gqaflashq8_0fma";
    public const string ValGqaFlashQ8_0Scalar = "gqaflashq8_0scalar";
    public const string ValMqaFlashQ8_0Avx2 = "mqaflashq8_0avx2";
    public const string ValMqaFlashQ8_0Fma = "mqaflashq8_0fma";
    public const string ValMqaFlashQ8_0Scalar = "mqaflashq8_0scalar";

    // Q4_0 Attention Values
    public const string ValMhaFlashQ4_0Avx2 = "mhaflashq4_0avx2";
    public const string ValMhaFlashQ4_0Fma = "mhaflashq4_0fma";
    public const string ValMhaFlashQ4_0Scalar = "mhaflashq4_0scalar";
    public const string ValGqaFlashQ4_0Avx2 = "gqaflashq4_0avx2";
    public const string ValGqaFlashQ4_0Fma = "gqaflashq4_0fma";
    public const string ValGqaFlashQ4_0Scalar = "gqaflashq4_0scalar";
    public const string ValMqaFlashQ4_0Avx2 = "mqaflashq4_0avx2";
    public const string ValMqaFlashQ4_0Fma = "mqaflashq4_0fma";
    public const string ValMqaFlashQ4_0Scalar = "mqaflashq4_0scalar";

    // Ffn Values
    public const string ValFfnDense = "dense";
    public const string ValFfnGated = "gated";
    public const string ValFfnMoE = "moe";

    // Norm Values
    public const string ValNormRMS = "rmsnorm";
    public const string ValNormRMSAvx2 = "rmsnormavx2";
    public const string ValNormRMSScalar = "rmsnormscalar";
    public const string ValNormLayer = "layernorm";
    public const string ValNormLayerAvx2 = "layernormavx2";
    public const string ValNormLayerScalar = "layernormscalar";

    // Linear Layer Keys
    public const string KeyLinear = "linear";
    public const string ValLinearAuto = "auto";
    public const string ValLinearFloat = "float";

    // Arch Values
    public const string ValDecoder = "decoder";
    public const string ValEncoder = "encoder";

    public ActivationKind Activation { get; init; } = ActivationKind.GELU;
    public GateKind       Gate       { get; init; } = GateKind.None;
    public FfnKind Ffn { get; init; } = FfnKind.Dense;
    public AttentionKind Attention { get; init; } = AttentionKind.MHA;
    public NormKind Norm { get; init; } = NormKind.RMSNorm;
    public ArchKind Arch { get; init; } = ArchKind.Decoder;
    public HardwareTier   Hardware   { get; init; } = HardwareTier.Auto;
    public bool Parallel { get; init; } = true;
    public bool FlashAttention { get; init; } = true;
    public bool UseHooks { get; init; } = false;

    public static SharpMindConfig Gpt => new()
    {
        Activation = ActivationKind.GELU,
        Gate = GateKind.None,
        Ffn = FfnKind.Dense,
        Attention = AttentionKind.MHA,
        Norm = NormKind.LayerNorm,
        Arch = ArchKind.Decoder,
    };

    public static SharpMindConfig Llama => new()
    {
        Activation = ActivationKind.SiLU,
        Gate = GateKind.SwiGLU,
        Ffn = FfnKind.Gated,
        Attention = AttentionKind.GQA,
        Norm = NormKind.RMSNorm,
        Arch = ArchKind.Decoder,
    };

    public static SharpMindConfig Bert => new()
    {
        Activation = ActivationKind.GELU,
        Gate = GateKind.None,
        Ffn = FfnKind.Dense,
        Attention = AttentionKind.MHA,
        Norm = NormKind.LayerNorm,
        Arch = ArchKind.Encoder,
    };
    public static SharpMindConfig Qwen => new()
    {
        Activation = ActivationKind.SiLU,
        Gate = GateKind.SwiGLU,
        Ffn = FfnKind.Gated,
        Attention = AttentionKind.MQA,
        Norm = NormKind.RMSNorm,
        Arch = ArchKind.Decoder,
    };
   
    public HardwareTier ResolvedHardware => Hardware switch
    {
        HardwareTier.Auto => HardwareTierHelpers.DetectBestTier(),
        _ => Hardware
    };


    /// <summary>
    /// Creates the correct <see cref="SharpMindConfig"/> for a model by reading
    /// model dimensions and an optional architecture name.
    /// The attention kind is always derived from NumHeads vs NumKvHeads.
    /// The <paramref name="architecture"/> string (e.g. "qwen2", "llama", "bert")
    /// selects the correct activation, gate, FFN, norm, and arch-type preset.
    /// </summary>
    public static SharpMindConfig ForModel(
        int numHeads, int numKvHeads,
        string? architecture = null,
        HardwareTier hw = HardwareTier.Auto)
    {
        var attn = numKvHeads switch
        {
            1 => AttentionKind.MQA,
            _ when numKvHeads == numHeads => AttentionKind.MHA,
            _ => AttentionKind.GQA
        };

        var (activation, gate, ffn, norm, arch) = architecture?.ToLowerInvariant() switch
        {
            "bert"                                                        => (ActivationKind.GELU,    GateKind.None,   FfnKind.Dense, NormKind.LayerNorm, ArchKind.Encoder),
            "gpt2" or "gptj" or "falcon" or "starcoder" or "starcoder2"
                or "bloom" or "phi" or "phi2"                            => (ActivationKind.GELU,    GateKind.None,   FfnKind.Dense, NormKind.LayerNorm, ArchKind.Decoder),
            "opt"                                                         => (ActivationKind.ReLU,    GateKind.None,   FfnKind.Dense, NormKind.LayerNorm, ArchKind.Decoder),
            "gemma"                                                       => (ActivationKind.GELU,    GateKind.GeGLU,  FfnKind.Gated, NormKind.RMSNorm,   ArchKind.Decoder),
            "mixtral" or "qwen2moe" or "deepseek2" or "dbrx"            => (ActivationKind.SiLU,    GateKind.SwiGLU, FfnKind.MoE,   NormKind.RMSNorm,   ArchKind.Decoder),
            "qwen3"                                                       => (ActivationKind.SiLU,    GateKind.SwiGLU, FfnKind.Gated, NormKind.RMSNorm,   ArchKind.Decoder),
            _                                                              => (ActivationKind.SiLU,    GateKind.SwiGLU, FfnKind.Gated, NormKind.RMSNorm,   ArchKind.Decoder),
        };

        return new SharpMindConfig
        {
            Activation = activation,
            Gate = gate,
            Ffn = ffn,
            Attention = attn,
            Norm = norm,
            Arch = arch,
            Hardware = hw
        };
    }

    public Dictionary<string, string> ToJigSawMapping(Dictionary<string, string>? overrides = null)
    {
        var cfg = new MappingBuilder(ResolvedHardware)
            .ApplyPreset(this)
            .ApplyQuantPreset(this)
            .Build();
        if (overrides == null) return cfg;
        if (overrides != null)
        {
            foreach (var m in overrides)
            {
                if (cfg.TryGetValue(m.Key, out string? value)) cfg[m.Key] = value;
                else cfg.Add(m.Key, m.Value);
            }
        }
        return cfg;
    }
}
