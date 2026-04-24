using System.Runtime.Intrinsics.X86;

namespace SharpMind;

public enum ActivationKind { GELU, SiLU, ReLU }
public enum GateKind       { None, SwiGLU, GeGLU }

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

    // ── Kernel values ─────────────────────────────────────────────────────
    public const string MapActivationKernelReLUAVX2 = "relu_avx2";
    public const string MapActivationKernelReLUScalar = "relu_scalar";
    public const string MapActivationKernelGELUScalar = "gelu_scalar";
    public const string MapActivationKernelGELUAVX2 = "gelu_avx2";
    public const string MapActivationKernelSiLUScalar = "silu_scalar";
    public const string MapActivationKernelSiLUAVX2 = "silu_avx2";
    public const string MapActivationKernelSwiGLUScalar = "swiglu_scalar";
    public const string MapActivationKernelSwiGLUAVX2 = "swiglu_avx2";
    public const string MapActivationKernelGeGLUScalar = "geglu_scalar";
    public const string MapActivationKernelGeGLUAVX2 = "geglu_avx2";
    public const string MapActivationKernelNoneScalar = "none_scalar";
    public const string MapActivationKernelNoneAVX2 = "none_avx2";
    public const string MapActivationKernelScalar = "scalar";
    public const string MapActivationKernelAVX2 = "avx2";
    public const string MapActivationKernelFMA = "fma";

    public ActivationKind Activation { get; init; } = ActivationKind.GELU;
    public GateKind       Gate       { get; init; } = GateKind.None;
    public HardwareTier   Hardware   { get; init; } = HardwareTier.Auto;

    // ── Pre-built presets ─────────────────────────────────────────────────

    /// <summary>GPT-2 / BERT: GELU, no gate.</summary>
    public static SharpMindConfig Gpt   => new() { Activation = ActivationKind.GELU, Gate = GateKind.None  };

    /// <summary>LLaMA 2/3 / Mistral: SiLU activation, SwiGLU gate.</summary>
    public static SharpMindConfig Llama => new() { Activation = ActivationKind.SiLU, Gate = GateKind.SwiGLU };

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
            [MapActivationKeyPointWise] = $"{act}_{hw}",
            [MapActivationKeyGate] = $"{gate}_{hw}",
            [MapActivationKeySoftMax] = hw,
            [MapActivationKeyRMSNorm] = hw,
            [MapActivationKeyMatMul] = MatMulHwKey,
        };
    }
}
