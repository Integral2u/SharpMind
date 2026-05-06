using System.Runtime.Intrinsics.X86;

namespace SharpMind;

public enum ActivationKind { GELU, SiLU, ReLU }
public enum GateKind       { None, SwiGLU, GeGLU }
public enum FfnKind { Dense, Gated, MoE }
public enum AttentionKind { MHA, GQA, MQA }
public enum NormKind { RMSNorm, LayerNorm }
public enum ArchKind { Decoder, Encoder }

public enum HardwareTier   { Auto, FMA, AVX2, Scalar }

public sealed record SharpMindConfig
{
    // ── JigSaw Pointers (Abstract method names) ────────────────────────────
    public const string PtrPointWise = "ApplyPointwise";
    public const string PtrGate = "ApplyGate";
    public const string PtrRMSNorm = "ApplyRMSNormRow";
    public const string PtrSoftmax = "ApplySoftmaxRow";
    public const string PtrMatMul = "MatMulInner";
    public const string PtrAttention = "ScaledDotProduct";
    public const string PtrFfn = "ApplyFfn";
    public const string PtrNorm = "ApplyRow";
    public const string PtrArch = "Forward";

    // ── Activation Keys ────────────────────────────────────────────────────────
    public const string KeyPointWise = "pointwise";
    public const string KeyGate = "gate";
    public const string KeyRMSNorm = "rmsnorm";
    public const string KeySoftmax = "softmax";
    public const string KeyMatMul = "matmul";
    // ── Model Layer Keys ────────────────────────────────────────────────────────
    public const string KeyAttention = "attention";
    public const string KeyFfn = "ffn";
    public const string KeyNorm = "norm";
    public const string KeyArch = "arch";
    // ── Training Keys ────────────────────────────────────────────────────────
    public const string KeyAdamW = "adamw";
    public const string KeyGradNorm = "gradnorm";

    // ── Activation Values ──────────────────────────────────────────────────────
    public const string ValReLU = "relu";
    public const string ValReLUAvx2 = "reluavx2";
    public const string ValGELU = "gelu";
    public const string ValGELUAvx2 = "geluavx2";
    public const string ValSiLU = "silu";
    public const string ValSiLUAvx2 = "siluavx2";
    public const string ValSwiGLU = "swiglu";
    public const string ValSwiGLUAvx2 = "swigluavx2";
    public const string ValGeGLU = "geglu";
    public const string ValGeGLUAvx2 = "gegluavx2";
    public const string ValNone = "none";
    public const string ValNoneAvx2 = "noneavx2";

    // ── Hardware Values ──────────────────────────────────────────────────────
    public const string ValScalar = "scalar";
    public const string ValAvx2 = "avx2";
    public const string ValFma = "fma";

    // ── Attention Values ──────────────────────────────────────────────────────
    public const string ValMhaAvx2 = "mhaavx2";
    public const string ValMhaScalar = "mhascalar";
    public const string ValGqaAvx2 = "gqaavx2";
    public const string ValGqaScalar = "gqascalar";
    public const string ValMqaAvx2 = "mqaavx2";
    public const string ValMqaScalar = "mqascalar";

    // ── Ffn Values ──────────────────────────────────────────────────────
    public const string ValFfnDense = "dense";
    public const string ValFfnGated = "gated";
    public const string ValFfnMoE = "moe";

    // ── Norm Values ──────────────────────────────────────────────────────
    public const string ValNormRMS = "rmsnorm";
    public const string ValNormRMSAvx2 = "rmsnormavx2";
    public const string ValNormRMSScalar = "rmsnormscalar";
    public const string ValNormLayer = "layernorm";
    public const string ValNormLayerAvx2 = "layernormavx2";
    public const string ValNormLayerScalar = "layernormscalar";

    // ── Fused Kernel Values ─────────────────────────────────────────────
    public const string KeyFusedNormLinear = "fusednormlinear";
    public const string ValFusedNormLinearAVX2 = "fusednormlinearavx2";
    public const string ValFusedNormLinearScalar = "fusednormlinearscalar";

    // ── Arch Values ──────────────────────────────────────────────────────
    public const string ValDecoder = "decoder";
    public const string ValEncoder = "encoder";

    public ActivationKind Activation { get; init; } = ActivationKind.GELU;
    public GateKind       Gate       { get; init; } = GateKind.None;
    public FfnKind Ffn { get; init; } = FfnKind.Dense;
    public AttentionKind Attention { get; init; } = AttentionKind.MHA;
    public NormKind Norm { get; init; } = NormKind.RMSNorm;
    public ArchKind Arch { get; init; } = ArchKind.Decoder;
    public HardwareTier   Hardware   { get; init; } = HardwareTier.Auto;

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

    public HardwareTier ResolvedHardware => Hardware switch
    {
        HardwareTier.Auto => Fma.IsSupported ? HardwareTier.FMA :
                             Avx2.IsSupported ? HardwareTier.AVX2 :
                                                 HardwareTier.Scalar,
        _ => Hardware
    };

    public Dictionary<string, string> ToJigSawMapping()
    {
        return new MappingBuilder(ResolvedHardware)
            .ApplyPreset(this)
            .Build();
    }
}