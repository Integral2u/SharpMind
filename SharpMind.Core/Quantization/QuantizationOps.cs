using System.IO;
using JigSawDotNet;
using System.Runtime.Intrinsics;

namespace SharpMind.Core.Quantization;

public abstract class QuantizationOps
{
    private const string NS  = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(Quantization)}.{nameof(QuantizationKernels)}";
    private const string MH  = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(MathHelpers)}";

    

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

    

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ8_0, true, null,
        "q8_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_FMA)}",
        "q8_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_AVX2)}",
        "q8_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_SSE)}",
        "q8_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_Scalar)}")]
    public abstract unsafe float VecDotQ8_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ8_0, true, null,
        "qmatmul_q8_0_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA)}",
        "qmatmul_q8_0_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_FMA)}",
        "qmatmul_q8_0_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_AVX2)}",
        "qmatmul_q8_0_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_AVX2)}",
        "qmatmul_q8_0_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar)}",
        "qmatmul_q8_0_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar)}",
        "qmatmul_q8_0_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar)}",
        "qmatmul_q8_0_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ8_0(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ5_0, true, null,
        "qmatmul_q5_0_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_FMA)}",
        "qmatmul_q5_0_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_FMA)}",
        "qmatmul_q5_0_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_AVX2)}",
        "qmatmul_q5_0_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_AVX2)}",
        "qmatmul_q5_0_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar)}",
        "qmatmul_q5_0_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar)}",
        "qmatmul_q5_0_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar)}",
        "qmatmul_q5_0_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ5_0(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ6K, true, null,
        "qmatmul_q6k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_FMA)}",
        "qmatmul_q6k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_FMA)}",
        "qmatmul_q6k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_AVX2)}",
        "qmatmul_q6k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_AVX2)}",
        "qmatmul_q6k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar)}",
        "qmatmul_q6k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar)}",
        "qmatmul_q6k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar)}",
        "qmatmul_q6k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ6K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ4_0, true, null,
        "q4_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_Scalar)}",
        "q4_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_AVX2)}",
        "q4_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_SSE)}",
        "q4_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_Scalar)}")]
    public abstract unsafe float VecDotQ4_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ4_0, true, null,
        "qmatmul_q4_0_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_Scalar)}",
        "qmatmul_q4_0_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_Scalar)}",
        "qmatmul_q4_0_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_AVX2)}",
        "qmatmul_q4_0_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_AVX2)}",
        "qmatmul_q4_0_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_SSE)}",
        "qmatmul_q4_0_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_SSE)}",
        "qmatmul_q4_0_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_Scalar)}",
        "qmatmul_q4_0_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ4_0(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ4_1, true, null,
        "q4_1_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_Scalar)}",
        "q4_1_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_AVX2)}",
        "q4_1_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_SSE)}",
        "q4_1_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_Scalar)}")]
    public abstract unsafe float VecDotQ4_1(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ4_1, true, null,
        "qmatmul_q4_1_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar)}",
        "qmatmul_q4_1_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar)}",
        "qmatmul_q4_1_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_AVX2)}",
        "qmatmul_q4_1_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_AVX2)}",
        "qmatmul_q4_1_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_SSE)}",
        "qmatmul_q4_1_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_SSE)}",
        "qmatmul_q4_1_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar)}",
        "qmatmul_q4_1_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ4_1(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ5_0, true, null,
        "q5_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_FMA)}",
        "q5_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_AVX2)}",
        "q5_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_Scalar)}",
        "q5_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_Scalar)}")]
    public abstract unsafe float VecDotQ5_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationConfig.KeyVecDotQ5_1, true, null,
        "q5_1_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_FMA)}",
        "q5_1_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_AVX2)}",
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



    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ2K, true, null,
        "qmatmul_q2k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_FMA)}",
        "qmatmul_q2k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_FMA)}",
        "qmatmul_q2k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_AVX2)}",
        "qmatmul_q2k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_AVX2)}",
        "qmatmul_q2k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar)}",
        "qmatmul_q2k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar)}",
        "qmatmul_q2k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar)}",
        "qmatmul_q2k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ2K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ3K, true, null,
        "qmatmul_q3k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_FMA)}",
        "qmatmul_q3k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_FMA)}",
        "qmatmul_q3k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_AVX2)}",
        "qmatmul_q3k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_AVX2)}",
        "qmatmul_q3k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar)}",
        "qmatmul_q3k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar)}",
        "qmatmul_q3k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar)}",
        "qmatmul_q3k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ3K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ4K, true, null,
        "qmatmul_q4k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_FMA)}",
        "qmatmul_q4k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_FMA)}",
        "qmatmul_q4k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_AVX2)}",
        "qmatmul_q4k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_AVX2)}",
        "qmatmul_q4k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar)}",
        "qmatmul_q4k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar)}",
        "qmatmul_q4k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar)}",
        "qmatmul_q4k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ4K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ5K, true, null,
        "qmatmul_q5k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_FMA)}",
        "qmatmul_q5k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_FMA)}",
        "qmatmul_q5k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_AVX2)}",
        "qmatmul_q5k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_AVX2)}",
        "qmatmul_q5k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar)}",
        "qmatmul_q5k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar)}",
        "qmatmul_q5k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar)}",
        "qmatmul_q5k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ5K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ8K, true, null,
        "qmatmul_q8k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_FMA)}",
        "qmatmul_q8k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_FMA)}",
        "qmatmul_q8k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_AVX2)}",
        "qmatmul_q8k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_AVX2)}",
        "qmatmul_q8k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_SSE)}",
        "qmatmul_q8k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_SSE)}",
        "qmatmul_q8k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_Scalar)}",
        "qmatmul_q8k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ8K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ8_1, true, null,
        "qmatmul_q8_1_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_FMA)}",
        "qmatmul_q8_1_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_FMA)}",
        "qmatmul_q8_1_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_AVX2)}",
        "qmatmul_q8_1_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_AVX2)}",
        "qmatmul_q8_1_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_SSE)}",
        "qmatmul_q8_1_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_SSE)}",
        "qmatmul_q8_1_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_Scalar)}",
        "qmatmul_q8_1_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ8_1(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ5_1, true, null,
        "qmatmul_q5_1_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_FMA)}",
        "qmatmul_q5_1_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_FMA)}",
        "qmatmul_q5_1_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_AVX2)}",
        "qmatmul_q5_1_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_AVX2)}",
        "qmatmul_q5_1_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "qmatmul_q5_1_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}",
        "qmatmul_q5_1_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "qmatmul_q5_1_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ5_1(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulQ4_NL, true, null,
        "qmatmul_q4_nl_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "qmatmul_q4_nl_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}",
        "qmatmul_q4_nl_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_AVX2)}",
        "qmatmul_q4_nl_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_AVX2)}",
        "qmatmul_q4_nl_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "qmatmul_q4_nl_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}",
        "qmatmul_q4_nl_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "qmatmul_q4_nl_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ4_NL(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulF32, true, null,
        "qmatmul_f32_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "qmatmul_f32_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "qmatmul_f32_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "qmatmul_f32_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "qmatmul_f32_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "qmatmul_f32_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "qmatmul_f32_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "qmatmul_f32_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulF32(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationConfig.KeyQuantizedMatMulF16, true, null,
        "qmatmul_f16_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "qmatmul_f16_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "qmatmul_f16_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "qmatmul_f16_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "qmatmul_f16_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "qmatmul_f16_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "qmatmul_f16_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "qmatmul_f16_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulF16(float* input, byte* rawWeights, float* output, int M, int K, int N);



    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ8_0, true, null,
        "read_q8_0_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ8_0_Scalar)}",
        "read_q8_0_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ8_0_Scalar)}",
        "read_q8_0_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ8_0_Scalar)}",
        "read_q8_0_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ8_0_Scalar)}")]
    public abstract unsafe void ReadQ8_0(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ4_0, true, null,
        "read_q4_0_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_0_Scalar)}",
        "read_q4_0_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ4_0_Scalar)}",
        "read_q4_0_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_0_Scalar)}",
        "read_q4_0_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ4_0_Scalar)}")]
    public abstract unsafe void ReadQ4_0(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ4_1, true, null,
        "read_q4_1_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_1_Scalar)}",
        "read_q4_1_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ4_1_Scalar)}",
        "read_q4_1_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_1_Scalar)}",
        "read_q4_1_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ4_1_Scalar)}")]
    public abstract unsafe void ReadQ4_1(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ5_0, true, null,
        "read_q5_0_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_0_Scalar)}",
        "read_q5_0_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ5_0_Scalar)}",
        "read_q5_0_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_0_Scalar)}",
        "read_q5_0_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ5_0_Scalar)}")]
    public abstract unsafe void ReadQ5_0(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ5_1, true, null,
        "read_q5_1_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_1_Scalar)}",
        "read_q5_1_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ5_1_Scalar)}",
        "read_q5_1_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_1_Scalar)}",
        "read_q5_1_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ5_1_Scalar)}")]
    public abstract unsafe void ReadQ5_1(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ8_1, true, null,
        "read_q8_1_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ8_1_Scalar)}",
        "read_q8_1_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ8_1_Scalar)}",
        "read_q8_1_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ8_1_Scalar)}",
        "read_q8_1_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ8_1_Scalar)}")]
    public abstract unsafe void ReadQ8_1(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ4_NL, true, null,
        "read_q4_nl_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_NL_Scalar)}",
        "read_q4_nl_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ4_NL_Scalar)}",
        "read_q4_nl_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_NL_Scalar)}",
        "read_q4_nl_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ4_NL_Scalar)}")]
    public abstract unsafe void ReadQ4_NL(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ2K, true, null,
        "read_q2k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ2K_Scalar)}",
        "read_q2k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ2K_Scalar)}",
        "read_q2k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ2K_Scalar)}",
        "read_q2k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ2K_Scalar)}")]
    public abstract unsafe void ReadQ2K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ3K, true, null,
        "read_q3k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ3_K_Scalar)}",
        "read_q3k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ3_K_Scalar)}",
        "read_q3k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ3_K_Scalar)}",
        "read_q3k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ3_K_Scalar)}")]
    public abstract unsafe void ReadQ3K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ4K, true, null,
        "read_q4k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ4K_Scalar)}",
        "read_q4k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ4K_Scalar)}",
        "read_q4k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ4K_Scalar)}",
        "read_q4k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ4K_Scalar)}")]
    public abstract unsafe void ReadQ4K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ5K, true, null,
        "read_q5k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_K_Scalar)}",
        "read_q5k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ5_K_Scalar)}",
        "read_q5k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_K_Scalar)}",
        "read_q5k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ5_K_Scalar)}")]
    public abstract unsafe void ReadQ5K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ6K, true, null,
        "read_q6k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ6K_Scalar)}",
        "read_q6k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ6K_Scalar)}",
        "read_q6k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ6K_Scalar)}",
        "read_q6k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ6K_Scalar)}")]
    public abstract unsafe void ReadQ6K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationConfig.KeyReadQ8K, true, null,
        "read_q8k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ8K_Scalar)}",
        "read_q8k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ8K_Scalar)}",
        "read_q8k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ8K_Scalar)}",
        "read_q8k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ8K_Scalar)}")]
    public abstract unsafe void ReadQ8K(BinaryReader reader, Span<float> data, int n);



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
