using System.Runtime.Intrinsics.X86;

namespace SharpMind;

public enum ActivationKind { GELU, SiLU, ReLU }
public enum GateKind       { None, SwiGLU, GeGLU }
public enum FfnKind { Dense, Gated, MoE }
public enum AttentionKind { MHA, GQA, MQA }
public enum NormKind { RMSNorm, LayerNorm }
public enum ArchKind { Decoder, Encoder }
/// <summary>
/// Hardware tier for SIMD kernel selection.
/// <see cref="Auto"/> detects at factory time — the check happens once,
/// never inside a hot kernel path.
/// </summary>
public enum HardwareTier   { Auto, FMA, AVX2, Scalar }

/// <summary>
/// Immutable model configuration. Passed to all factories to determine
/// which JigSaw mapping is assembled.
/// </summary>
public sealed record SharpMindConfig
{
    // ── Mapping keys (slot names passed to JigSaw) ────────────────────────
    public const string MapActivationKeyPointWise = "pointwise";
    public const string MapActivationKeyGate = "gate";
    public const string MapActivationKeyRMSNorm = "rmsnorm";
    public const string MapActivationKeySoftMax = "softmax";
    public const string MapActivationKeyMatMul = "matmul";
    public const string MapAttentionKey = "attention";
    public const string MapFfnKey = "ffn";
    public const string MapNormKey = "norm";
    public const string MapArchKey = "arch";

    // ── Kernel values ─────────────────────────────────────────────────────
    public const string MapActivationKernelReLUAVX2 = "reluavx2";
    public const string MapActivationKernelReLUScalar = "reluscalar";
    public const string MapActivationKernelGELUScalar = "geluscalar";
    public const string MapActivationKernelGELUAVX2 = "geluavx2";
    public const string MapActivationKernelSiLUScalar = "siluscalar";
    public const string MapActivationKernelSiLUAVX2 = "siluavx2";
    public const string MapActivationKernelSwiGLUScalar = "swigluscalar";
    public const string MapActivationKernelSwiGLUAVX2 = "swigluavx2";
    public const string MapActivationKernelGeGLUScalar = "gegluscalar";
    public const string MapActivationKernelGeGLUAVX2 = "gegluavx2";
    public const string MapActivationKernelNoneScalar = "nonescalar";
    public const string MapActivationKernelNoneAVX2 = "noneavx2";
    public const string MapActivationKernelScalar = "scalar";
    public const string MapActivationKernelAVX2 = "avx2";
    public const string MapActivationKernelFMA = "fma";

    // ── Kernel values — attention (variant_hw) ────────────────────────────
    public const string MapAttentionKernelMhaAvx2 = "mhaavx2";
    public const string MapAttentionKernelMhaScalar = "mhascalar";
    public const string MapAttentionKernelGqaAvx2 = "gqaavx2";
    public const string MapAttentionKernelGqaScalar = "gqascalar";
    public const string MapAttentionKernelMqaAvx2 = "mqaavx2";
    public const string MapAttentionKernelMqaScalar = "mqascalar";

    // ── Kernel values — ffn ───────────────────────────────────────────────
    public const string KernelFfnDense = "dense";
    public const string KernelFfnGated = "gated";
    public const string KernelFfnMoE = "moe";

    // ── Kernel values — norm ──────────────────────────────────────────────
    public const string KernelNormRMS = "rmsnorm";
    public const string KernelNormRMSAVX2 = "rmsnormavx2";
    public const string KernelNormRMSScalar = "rmsnormscalar";
    public const string KernelNormLayer = "layernorm";
    public const string KernelNormLayerAVX2 = "layernormavx2";
    public const string KernelNormLayerScalar = "layernormscalar";

    // ── Kernel values — arch ──────────────────────────────────────────────
    public const string KernelDecoder = "decoder";
    public const string KernelEncoder = "encoder";

    public ActivationKind Activation { get; init; } = ActivationKind.GELU;
    public GateKind       Gate       { get; init; } = GateKind.None;
    public FfnKind Ffn { get; init; } = FfnKind.Dense;
    public AttentionKind Attention { get; init; } = AttentionKind.MHA;
    public NormKind Norm { get; init; } = NormKind.RMSNorm;
    public ArchKind Arch { get; init; } = ArchKind.Decoder;
    public HardwareTier   Hardware   { get; init; } = HardwareTier.Auto;

    // ── Pre-built presets ─────────────────────────────────────────────────

    /// <summary>GPT-2: GELU, dense FFN, MHA, LayerNorm, decoder.</summary>
    public static SharpMindConfig Gpt => new()
    {
        Activation = ActivationKind.GELU,
        Gate = GateKind.None,
        Ffn = FfnKind.Dense,
        Attention = AttentionKind.MHA,
        Norm = NormKind.LayerNorm,
        Arch = ArchKind.Decoder,
    };

    /// <summary>LLaMA 2/3 / Mistral: SiLU, SwiGLU gated FFN, GQA, RMSNorm, decoder.</summary>
    public static SharpMindConfig Llama => new()
    {
        Activation = ActivationKind.SiLU,
        Gate = GateKind.SwiGLU,
        Ffn = FfnKind.Gated,
        Attention = AttentionKind.GQA,
        Norm = NormKind.RMSNorm,
        Arch = ArchKind.Decoder,
    };

    /// <summary>BERT: GELU, dense FFN, MHA, LayerNorm, encoder.</summary>
    public static SharpMindConfig Bert => new()
    {
        Activation = ActivationKind.GELU,
        Gate = GateKind.None,
        Ffn = FfnKind.Dense,
        Attention = AttentionKind.MHA,
        Norm = NormKind.LayerNorm,
        Arch = ArchKind.Encoder,
    };
    
    // ── JigSaw mapping ────────────────────────────────────────────────────

    /// <summary>
    /// Resolves hardware tier once. FMA implies AVX2 on all x86 CPUs.
    /// </summary>
    public HardwareTier ResolvedHardware => Hardware switch
    {
        HardwareTier.Auto => Fma.IsSupported ? HardwareTier.FMA :
                             Avx2.IsSupported ? HardwareTier.AVX2 :
                                                HardwareTier.Scalar,
        _ => Hardware
    };
    /// <summary>
    /// Activation hw key — FMA maps to avx2 because exp/tanh kernels
    /// get no benefit from the fused multiply-add instruction.
    /// </summary>
    private string ActHwKey => ResolvedHardware == HardwareTier.Scalar ? MapActivationKernelScalar : MapActivationKernelAVX2;

    /// <summary>
    /// MatMul hw key — FMA is a genuine third path, giving measurably
    /// better throughput than plain AVX2 on Haswell+ CPUs.
    /// </summary>
    private string MatMulHwKey => ResolvedHardware switch
    {
        HardwareTier.FMA => MapActivationKernelFMA,
        HardwareTier.AVX2 => MapActivationKernelAVX2,
        _ => MapActivationKernelScalar
    };
    /// <summary>
    /// Builds the complete JigSaw mapping. All values are resolved
    /// deterministically here — no runtime hardware checks remain after this call.
    /// </summary>
    public Dictionary<string, string> ToJigSawMapping()
    {
        string hw = ActHwKey;
        string act = Activation.ToString().ToLowerInvariant();
        string gate = Gate.ToString().ToLowerInvariant();

        return new Dictionary<string, string>
        {
            // ── Activation kernels ──────────────────────────────────────
            [MapActivationKeyPointWise] = $"{act}{hw}",
            [MapActivationKeyGate] = $"{gate}{hw}",
            [MapActivationKeySoftMax] = hw,
            [MapActivationKeyRMSNorm] = hw,
            [MapActivationKeyMatMul] = MatMulHwKey,

            // ── Model layer kernels ─────────────────────────────────────
            [MapAttentionKey] = $"{Attention.ToString().ToLowerInvariant()}{hw}",
            [MapFfnKey] = Ffn.ToString().ToLowerInvariant(),
            [MapNormKey] = Norm == NormKind.RMSNorm ? KernelNormRMS : KernelNormLayer,
            [MapArchKey] = Arch == ArchKind.Decoder ? KernelDecoder : KernelEncoder,
        };
    }
}
