using JigSawDotNet;
using System.Runtime.Intrinsics;

namespace SharpMind.Core.Quantization;

public abstract class QuantizationOps
{
    private const string NS  = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(Quantization)}.{nameof(QuantizationKernels)}";
    private const string MH  = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(MathHelpers)}";

    
    // K-Quant VecDot methods (QK_K=256)
    

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ3K, true, null,
        "q3k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ3K_FMA)}",
        "q3k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ3K_AVX2)}",
        "q3k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ3K_Scalar)}",
        "q3k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ3K_Scalar)}")]
    public abstract unsafe float VecDotQ3K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ4K, true, null,
        "q4k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_FMA)}",
        "q4k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_AVX2)}",
        "q4k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_Scalar)}",
        "q4k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_Scalar)}")]
    public abstract unsafe float VecDotQ4K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ5K, true, null,
        "q5k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5K_FMA)}",
        "q5k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ5K_AVX2)}",
        "q5k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5K_Scalar)}",
        "q5k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ5K_Scalar)}")]
    public abstract unsafe float VecDotQ5K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ6K, true, null,
        "q6k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ6K_FMA)}",
        "q6k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ6K_AVX2)}",
        "q6k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ6K_Scalar)}",
        "q6k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ6K_Scalar)}")]
    public abstract unsafe float VecDotQ6K(float* input, byte* rawWeights, int col, int inFeatures);

    
    // Simple-block VecDot methods (QK=32)
    

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ8_0, true, null,
        "q8_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_FMA)}",
        "q8_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_AVX2)}",
        "q8_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_SSE)}",
        "q8_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_Scalar)}")]
    public abstract unsafe float VecDotQ8_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ8_0, true, null,
        "qmatmul_q8_0_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_FMA)}",
        "qmatmul_q8_0_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_AVX2)}",
        "qmatmul_q8_0_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Scalar)}",
        "qmatmul_q8_0_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ8_0(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ5_0, true, null,
        "qmatmul_q5_0_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_FMA)}",
        "qmatmul_q5_0_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_AVX2)}",
        "qmatmul_q5_0_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Scalar)}",
        "qmatmul_q5_0_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ5_0(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ6K, true, null,
        "qmatmul_q6k_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_FMA)}",
        "qmatmul_q6k_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_AVX2)}",
        "qmatmul_q6k_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Scalar)}",
        "qmatmul_q6k_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ6K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ4_0, true, null,
        "q4_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_Scalar)}",
        "q4_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_AVX2)}",
        "q4_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_SSE)}",
        "q4_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_Scalar)}")]
    public abstract unsafe float VecDotQ4_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ4_1, true, null,
        "q4_1_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_Scalar)}",
        "q4_1_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_AVX2)}",
        "q4_1_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_SSE)}",
        "q4_1_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_Scalar)}")]
    public abstract unsafe float VecDotQ4_1(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ5_0, true, null,
        "q5_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_Scalar)}",
        "q5_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_Scalar)}",
        "q5_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_Scalar)}",
        "q5_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_Scalar)}")]
    public abstract unsafe float VecDotQ5_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ5_1, true, null,
        "q5_1_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_Scalar)}",
        "q5_1_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_Scalar)}",
        "q5_1_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_Scalar)}",
        "q5_1_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_Scalar)}")]
    public abstract unsafe float VecDotQ5_1(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ4_NL,
        "q4_nl_fma",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_NL_AVX2)}",
        "q4_nl_avx2",  $"{NS}.{nameof(QuantizationKernels.VecDotQ4_NL_AVX2)}",
        "q4_nl_sse",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_NL_Scalar)}",
        "q4_nl_scalar",$"{NS}.{nameof(QuantizationKernels.VecDotQ4_NL_Scalar)}")]
    public abstract unsafe float VecDotQ4_NL(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ8_1, true, null,
        "q8_1_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_1_FMA)}",
        "q8_1_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ8_1_AVX2)}",
        "q8_1_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_1_SSE)}",
        "q8_1_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ8_1_Scalar)}")]
    public abstract unsafe float VecDotQ8_1(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ2K, true, null,
        "q2k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ2K_FMA)}",
        "q2k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ2K_AVX2)}",
        "q2k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ2K_Scalar)}",
        "q2k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ2K_Scalar)}")]
    public abstract unsafe float VecDotQ2K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ8K, true, null,
        "q8k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8K_FMA)}",
        "q8k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ8K_AVX2)}",
        "q8k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8K_SSE)}",
        "q8k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ8K_Scalar)}")]
    public abstract unsafe float VecDotQ8K(float* input, byte* rawWeights, int col, int inFeatures);

    
    // Shared helpers
    

    [PuzzleCornerPiece(QuantizationConfig.KeyHSum256, true, null,
        "hsum_fma",   $"{MH}.{nameof(MathHelpers.HSum256_Avx)}",
        "hsum_avx2",  $"{MH}.{nameof(MathHelpers.HSum256_Avx)}",
        "hsum_sse3",  $"{MH}.{nameof(MathHelpers.HSum256_Sse3)}",
        "hsum_sse",   $"{MH}.{nameof(MathHelpers.HSum256_Sse3)}",
        "hsum_scalar", $"{MH}.{nameof(MathHelpers.HSum256_Scalar)}")]
    public abstract float HSum256(Vector256<float> v);

    [PuzzleCornerPiece(QuantizationConfig.KeyHalfToFloat, true, null,
        "halftofloat_fma",   $"{NS}.{nameof(QuantizationKernels.HalfToFloat_F16C)}",
        "halftofloat_avx2",  $"{NS}.{nameof(QuantizationKernels.HalfToFloat_F16C)}",
        "halftofloat_sse",   $"{NS}.{nameof(QuantizationKernels.HalfToFloat_Scalar)}",
        "halftofloat_scalar", $"{NS}.{nameof(QuantizationKernels.HalfToFloat_Scalar)}")]
    public abstract float HalfToFloat(ushort half);

    [PuzzleCornerPiece(QuantizationConfig.KeyFloatToHalf, true, null,
        "floattohalf_fma",   $"{NS}.{nameof(QuantizationKernels.FloatToHalf_F16C)}",
        "floattohalf_avx2",  $"{NS}.{nameof(QuantizationKernels.FloatToHalf_F16C)}",
        "floattohalf_sse",   $"{NS}.{nameof(QuantizationKernels.FloatToHalf_Scalar)}",
        "floattohalf_scalar", $"{NS}.{nameof(QuantizationKernels.FloatToHalf_Scalar)}")]
    public abstract ushort FloatToHalf(float f);

    [PuzzleCornerPiece(QuantizationConfig.KeyGetScaleMinK4_Scale, true, null,
        "getscalemink4_scale_fma",    $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Scale_Scalar)}",
        "getscalemink4_scale_avx2",   $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Scale_Scalar)}",
        "getscalemink4_scale_sse",    $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Scale_Scalar)}",
        "getscalemink4_scale_scalar", $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Scale_Scalar)}")]
    public abstract unsafe byte GetScaleMinK4_Scale(int j, byte* scales);

    [PuzzleCornerPiece(QuantizationConfig.KeyGetScaleMinK4_Min, true, null,
        "getscalemink4_min_fma",    $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Min_Scalar)}",
        "getscalemink4_min_avx2",   $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Min_Scalar)}",
        "getscalemink4_min_sse",    $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Min_Scalar)}",
        "getscalemink4_min_scalar", $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Min_Scalar)}")]
    public abstract unsafe byte GetScaleMinK4_Min(int j, byte* scales);
}
