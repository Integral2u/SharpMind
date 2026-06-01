using JigSawDotNet;

namespace SharpMind.Inference;

// ─────────────────────────────────────────────────────────────────────────────
// InferenceOps — abstract, JigSaw-assembled
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// JigSaw-assembled inference kernels. Three slots:
///
///   "decode_attn"   — single-token decode attention against KV cache
///                     standard vs flash × avx2 vs scalar
///   "prefill_attn"  — full-sequence prefill attention
///                     standard vs flash × avx2 vs scalar
///   "quant_matmul"  — weight projection with optional INT8/INT4 dequantization
///                     fp32 vs int8 vs int4
///
/// The JigSaw key selects the algorithm AND the hardware path in one value,
/// allowing decode and prefill to use different algorithms independently.
/// </summary>
public abstract class InferenceOps
{

    private const string NS = $"{nameof(SharpMind)}.{nameof(Inference)}.{nameof(InferenceKernels)}";
    // ── Decode attention ──────────────────────────────────────────────────

    [PuzzleCornerPiece(InferenceConfig.PtrDecodeAttention,
        InferenceConfig.ValStandardAvx2,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Standard_AVX2),
        InferenceConfig.ValStandardFma,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Standard_FMA),
        InferenceConfig.ValStandardScalar,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Standard_Scalar),
        InferenceConfig.ValFlashAvx2,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Flash_AVX2),
        InferenceConfig.ValFlashFma,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Flash_FMA),
        InferenceConfig.ValFlashScalar,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Flash_Scalar))]
    public abstract unsafe void DecodeAttention(
        float* q, float* k, float* v, float* output,
        int cacheLen, int headDim, float scale);

    // ── Prefill attention ─────────────────────────────────────────────────
    // Same kernel options — prefill and decode can be configured independently

    [PuzzleCornerPiece(InferenceConfig.PtrPrefillAttention,
        InferenceConfig.ValStandardAvx2,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Standard_AVX2),
        InferenceConfig.ValStandardFma,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Standard_FMA),
        InferenceConfig.ValStandardScalar,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Standard_Scalar),
        InferenceConfig.ValFlashAvx2,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Flash_AVX2),
        InferenceConfig.ValFlashFma,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Flash_FMA),
        InferenceConfig.ValFlashScalar,
            NS + "." + nameof(InferenceKernels.DecodeAttention_Flash_Scalar))]
    public abstract unsafe void PrefillAttention(
        float* q, float* k, float* v, float* output,
        int cacheLen, int headDim, float scale);

    // ── Quantized matmul ──────────────────────────────────────────────────

    private const string QuantNS = $"{nameof(SharpMind)}.{nameof(Inference)}.{nameof(QuantKernels)}";

    [PuzzleCornerPiece(InferenceConfig.PtrQuantMatMul,
        InferenceConfig.ValQuantNone,
            NS + "." + nameof(InferenceKernels.QuantMatMul_FP32),
        InferenceConfig.ValQuantInt8,
            NS + "." + nameof(InferenceKernels.QuantMatMul_Int8),
        InferenceConfig.ValQuantInt4,
            QuantNS + "." + nameof(QuantKernels.QuantMatMul_Int4),
        InferenceConfig.ValQuantInt2,
            QuantNS + "." + nameof(QuantKernels.QuantMatMul_Int2),
        InferenceConfig.ValQuantInt1,
            QuantNS + "." + nameof(QuantKernels.QuantMatMul_Int1),
        InferenceConfig.ValQuantTernary,
            QuantNS + "." + nameof(QuantKernels.QuantMatMul_Ternary),
        InferenceConfig.ValQuantFP8,
            QuantNS + "." + nameof(QuantKernels.QuantMatMul_FP8))]
    public abstract unsafe void QuantMatMul(
        float* input, float* weights, float* output,
        float* scales, int inFeatures, int outFeatures);
}
