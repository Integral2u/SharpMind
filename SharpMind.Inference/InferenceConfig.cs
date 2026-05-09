namespace SharpMind.Inference;

using SharpMind;

/// <summary>
/// Inference-specific kernel and execution configuration.
/// Extends <see cref="SharpMindConfig"/> with slots that only make sense
/// during inference — flash attention, quantization, decode mode.
/// </summary>
public enum AttentionAlgo { Standard, Flash }
public enum QuantKind     { None, Int8, Int4, Int2, Int1, Ternary, FP8 }
public enum BatchMode     { Single, Continuous }

public sealed record InferenceConfig
{
    // ── JigSaw Pointer names ──────────────────────────────────────────────
    public const string PtrDecodeAttention  = "DecodeAttention";
    public const string PtrPrefillAttention = "PrefillAttention";
    public const string PtrQuantMatMul      = "QuantMatMul";

    // ── JigSaw Keys ───────────────────────────────────────────────────────
    public const string KeyDecodeAttention  = "decode_attn";
    public const string KeyPrefillAttention = "prefill_attn";
    public const string KeyQuantMatMul      = "quant_matmul";

    // ── Values: attention algo ────────────────────────────────────────────
    public const string ValStandardAvx2  = "standardavx2";
    public const string ValStandardScalar = "standardscalar";
    public const string ValFlashAvx2     = "flashavx2";
    public const string ValFlashScalar   = "flashscalar";

    // ── Values: quantization ──────────────────────────────────────────────
    public const string ValQuantNone = "fp32";
    public const string ValQuantInt8 = "int8";
    public const string ValQuantInt4 = "int4";
    public const string ValQuantInt2 = "int2";
    public const string ValQuantInt1 = "int1";
    public const string ValQuantTernary = "ternary";
    public const string ValQuantFP8 = "fp8";

    // ── Config properties ─────────────────────────────────────────────────

    public AttentionAlgo Attention   { get; init; } = AttentionAlgo.Standard;
    public QuantKind     Quant       { get; init; } = QuantKind.None;
    public BatchMode     Batching    { get; init; } = BatchMode.Single;

    /// <summary>
    /// KV-cache sliding window size. 0 = no sliding window (full context).
    /// When set, the oldest tokens are evicted when the cache fills.
    /// </summary>
    public int SlidingWindowSize { get; init; } = 0;

    // ── Presets ───────────────────────────────────────────────────────────

    /// <summary>Standard float32 inference — maximum compatibility.</summary>
    public static InferenceConfig Default => new();

    /// <summary>Flash Attention float32 — faster prefill on long contexts.</summary>
    public static InferenceConfig Fast => new()
    {
        Attention = AttentionAlgo.Flash,
    };

    /// <summary>INT8 weight quantization with Flash Attention.</summary>
    public static InferenceConfig Quantized => new()
    {
        Attention = AttentionAlgo.Flash,
        Quant     = QuantKind.Int8,
    };

    /// <summary>INT4 weight quantization — 4× memory savings.</summary>
    public static InferenceConfig QuantizedInt4 => new()
    {
        Attention = AttentionAlgo.Flash,
        Quant     = QuantKind.Int4,
    };

    /// <summary>INT2 weight quantization — 8× memory savings.</summary>
    public static InferenceConfig QuantizedInt2 => new()
    {
        Attention = AttentionAlgo.Flash,
        Quant     = QuantKind.Int2,
    };

    /// <summary>Ternary (1.58-bit) quantization — ~16× memory savings.</summary>
    public static InferenceConfig QuantizedTernary => new()
    {
        Attention = AttentionAlgo.Flash,
        Quant     = QuantKind.Ternary,
    };

    /// <summary>FP8 (E4M3) quantization — 4× memory savings.</summary>
    public static InferenceConfig QuantizedFP8 => new()
    {
        Attention = AttentionAlgo.Flash,
        Quant     = QuantKind.FP8,
    };

    /// <summary>Continuous batching for serving multiple requests.</summary>
    public static InferenceConfig Serving => new()
    {
        Attention = AttentionAlgo.Flash,
        Batching  = BatchMode.Continuous,
    };

    // ── Mapping ───────────────────────────────────────────────────────────

    /// <summary>
    /// Builds the JigSaw mapping for inference ops.
    /// Merged with the base <see cref="SharpMindConfig.ToJigSawMapping"/>
    /// in <see cref="InferenceOpsFactory"/>.
    /// </summary>
    public Dictionary<string, string> ToJigSawMapping(HardwareTier hw)
    {
        string hwSuffix = hw == HardwareTier.Scalar ? "scalar" : "avx2";

        string attnVal = Attention switch
        {
            AttentionAlgo.Flash    => $"flash{hwSuffix}",
            _                      => $"standard{hwSuffix}",
        };

        return new Dictionary<string, string>
        {
            [PtrDecodeAttention]  = attnVal,
            [PtrPrefillAttention] = attnVal,
[PtrQuantMatMul] = Quant switch
            {
                QuantKind.Int8 => ValQuantInt8,
                QuantKind.Int4 => ValQuantInt4,
                QuantKind.Int2 => ValQuantInt2,
                QuantKind.Int1 => ValQuantInt1,
                QuantKind.Ternary => ValQuantTernary,
                QuantKind.FP8 => ValQuantFP8,
                _ => ValQuantNone,
            },
        };
    }
}
