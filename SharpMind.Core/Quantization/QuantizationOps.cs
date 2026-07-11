using JigSawDotNet;
using System.Data;
using System.IO;
using System.Runtime.Intrinsics;

namespace SharpMind.Core.Quantization;

public unsafe delegate float VecDotFn(float* input, byte* rawWeights, int col, int inFeatures);
public unsafe delegate void QuantizedMatMulFn(float* input, byte* rawWeights, float* output, int M, int K, int N);
public abstract class QuantizationOps
{
    private const string NS  = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(Quantization)}.{nameof(QuantizationKernels)}";
    private const string MH  = $"{nameof(SharpMind)}.{nameof(Core)}.{nameof(MathHelpers)}";

    public static long GetRawTensorByteCount(int[] shape, QuantDType dtype)
    {
        long totalElements = 1;
        foreach (int d in shape) totalElements *= d;

        return dtype switch
        {
            QuantDType.F32 => totalElements * 4,
            QuantDType.F16 => totalElements * 2,
            QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L => ((totalElements + 255) / 256) * 110,
            QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M => ((totalElements + 255) / 256) * 144,
            QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M => ((totalElements + 255) / 256) * 176,
            QuantDType.Q6_K or QuantDType.Q6_K_S => ((totalElements + 255) / 256) * 210,
            QuantDType.Q2_K or QuantDType.Q2_K_S => ((totalElements + 255) / 256) * 84,
            QuantDType.Q8_K => ((totalElements + 255) / 256) * 292,
            QuantDType.Q8_0 => ((totalElements + 31) / 32) * 34,
            QuantDType.Q8_1 => ((totalElements + 31) / 32) * 36,
            QuantDType.Q5_0 => ((totalElements + 31) / 32) * 22,
            QuantDType.Q5_1 => ((totalElements + 31) / 32) * 24,
            QuantDType.Q4_0 => ((totalElements + 31) / 32) * 18,
            QuantDType.IQ4_NL => ((totalElements + 31) / 32) * 18,
            QuantDType.Q4_1 => ((totalElements + 31) / 32) * 20,
            _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, null)
        };
    }
    public void ReadFor(QuantDType dType, BinaryReader reader, Span<float> data, int n)
    {
        switch (dType)
        {
            case QuantDType.F32:   ReadF32(reader, data, n); break;
            case QuantDType.F16:   ReadF16(reader, data, n); break;
            case QuantDType.Q4_0:  ReadQ4_0(reader, data, n); break;
            case QuantDType.Q4_1:  ReadQ4_1(reader, data, n); break;
            case QuantDType.Q5_0:  ReadQ5_0(reader, data, n); break;
            case QuantDType.Q5_1:  ReadQ5_1(reader, data, n); break;
            case QuantDType.Q8_0:  ReadQ8_0(reader, data, n); break;
            case QuantDType.Q8_1:  ReadQ8_1(reader, data, n); break;
            case QuantDType.IQ4_NL: ReadQ4_NL(reader, data, n); break;
            case QuantDType.Q2_K:
            case QuantDType.Q2_K_S:  ReadQ2K(reader, data, n); break;
            case QuantDType.Q3_K:
            case QuantDType.Q3_K_S:
            case QuantDType.Q3_K_M:
            case QuantDType.Q3_K_L:  ReadQ3K(reader, data, n); break;
            case QuantDType.Q4_K:
            case QuantDType.Q4_K_S:
            case QuantDType.Q4_K_M:  ReadQ4K(reader, data, n); break;
            case QuantDType.Q5_K:
            case QuantDType.Q5_K_S:
            case QuantDType.Q5_K_M:  ReadQ5K(reader, data, n); break;
            case QuantDType.Q6_K:
            case QuantDType.Q6_K_S:  ReadQ6K(reader, data, n); break;
            case QuantDType.Q8_K:    ReadQ8K(reader, data, n); break;
            default:
                throw new ArgumentOutOfRangeException(nameof(dType), dType, null);
        }
    }
    public unsafe float VecDotFor(QuantDType dType, float* input, byte* rawWeights, int col, int inFeatures) => VecDotOpFor(dType)(input, rawWeights, col, inFeatures);
    public unsafe VecDotFn VecDotOpFor(QuantDType dType) => dType switch
    {
        QuantDType.F32 => VecDotF32,
        QuantDType.F16 => VecDotF16,
        QuantDType.Q4_0 => VecDotQ4_0,
        QuantDType.Q4_1 => VecDotQ4_1,
        QuantDType.Q5_0 => VecDotQ5_0,
        QuantDType.Q5_1 => VecDotQ5_1,
        QuantDType.Q8_0 => VecDotQ8_0,
        QuantDType.Q8_1 => VecDotQ8_1,
        QuantDType.IQ4_NL => VecDotQ4_NL,
        QuantDType.Q2_K or QuantDType.Q2_K_S => VecDotQ2K,
        QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L => VecDotQ3K,
        QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M => VecDotQ4K,
        QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M => VecDotQ5K,
        QuantDType.Q6_K or QuantDType.Q6_K_S => VecDotQ6K,
        QuantDType.Q8_K => VecDotQ8K,
        _ => throw new ArgumentOutOfRangeException(nameof(dType), dType, null),
    };

    public unsafe void QuantizedMatMulFor(QuantDType dType, float* input, byte* rawWeights, float* output, int M, int K, int N) => QuantizedMatMulOpFor(dType)(input, rawWeights, output, M, K, N);

    public unsafe QuantizedMatMulFn QuantizedMatMulOpFor(QuantDType dType) => dType switch
    {
        QuantDType.F32 => QuantizedMatMulF32,
        QuantDType.F16 => QuantizedMatMulF16,
        QuantDType.Q4_0 => QuantizedMatMulQ4_0,
        QuantDType.Q4_1 => QuantizedMatMulQ4_1,
        QuantDType.Q5_0 => QuantizedMatMulQ5_0,
        QuantDType.Q5_1 => QuantizedMatMulQ5_1,
        QuantDType.Q8_0 => QuantizedMatMulQ8_0,
        QuantDType.Q8_1 => QuantizedMatMulQ8_1,
        QuantDType.IQ4_NL => QuantizedMatMulQ4_NL,
        QuantDType.Q2_K or QuantDType.Q2_K_S => QuantizedMatMulQ2K,
        QuantDType.Q3_K or QuantDType.Q3_K_S or QuantDType.Q3_K_M or QuantDType.Q3_K_L => QuantizedMatMulQ3K,
        QuantDType.Q4_K or QuantDType.Q4_K_S or QuantDType.Q4_K_M => QuantizedMatMulQ4K,
        QuantDType.Q5_K or QuantDType.Q5_K_S or QuantDType.Q5_K_M => QuantizedMatMulQ5K,
        QuantDType.Q6_K or QuantDType.Q6_K_S => QuantizedMatMulQ6K,
        QuantDType.Q8_K => QuantizedMatMulQ8K,
        _ => throw new ArgumentOutOfRangeException(nameof(dType), dType, null),
    };

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ3K, true, null,
        "q3k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ3K_FMA)}",
        "q3k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ3K_AVX2)}",
        "q3k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ3K_Scalar)}",
        "q3k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ3K_Scalar)}")]
    public abstract unsafe float VecDotQ3K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ4K, true, null,
        "q4k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_FMA)}",
        "q4k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_AVX2)}",
        "q4k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_Scalar)}",
        "q4k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4K_Scalar)}")]
    public abstract unsafe float VecDotQ4K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ5K, true, null,
        "q5k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5K_FMA)}",
        "q5k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ5K_AVX2)}",
        "q5k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5K_Scalar)}",
        "q5k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ5K_Scalar)}")]
    public abstract unsafe float VecDotQ5K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ6K, true, null,
        "q6k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ6K_FMA)}",
        "q6k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ6K_AVX2)}",
        "q6k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ6K_Scalar)}",
        "q6k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ6K_Scalar)}")]
    public abstract unsafe float VecDotQ6K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ8_0, true, null,
        "q8_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_FMA)}",
        "q8_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_AVX2)}",
        "q8_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_SSE)}",
        "q8_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ8_0_Scalar)}")]
    public abstract unsafe float VecDotQ8_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ8_0, true, null,
        "qmatmul_q8_0_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA)}",
        "qmatmul_q8_0_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_FMA)}",
        "qmatmul_q8_0_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_AVX2)}",
        "qmatmul_q8_0_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_AVX2)}",
        "qmatmul_q8_0_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar)}",
        "qmatmul_q8_0_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar)}",
        "qmatmul_q8_0_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar)}",
        "qmatmul_q8_0_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_0_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ8_0(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ5_0, true, null,
        "qmatmul_q5_0_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_FMA)}",
        "qmatmul_q5_0_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_FMA)}",
        "qmatmul_q5_0_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_AVX2)}",
        "qmatmul_q5_0_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_AVX2)}",
        "qmatmul_q5_0_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar)}",
        "qmatmul_q5_0_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar)}",
        "qmatmul_q5_0_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar)}",
        "qmatmul_q5_0_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_0_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ5_0(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ6K, true, null,
        "qmatmul_q6k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_FMA)}",
        "qmatmul_q6k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_FMA)}",
        "qmatmul_q6k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_AVX2)}",
        "qmatmul_q6k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_AVX2)}",
        "qmatmul_q6k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar)}",
        "qmatmul_q6k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar)}",
        "qmatmul_q6k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar)}",
        "qmatmul_q6k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ6K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ6K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ4_0, true, null,
        "q4_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_Scalar)}",
        "q4_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_AVX2)}",
        "q4_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_SSE)}",
        "q4_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4_0_Scalar)}")]
    public abstract unsafe float VecDotQ4_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ4_0, true, null,
        "qmatmul_q4_0_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_AVX2)}",
        "qmatmul_q4_0_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_AVX2)}",
        "qmatmul_q4_0_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_AVX2)}",
        "qmatmul_q4_0_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_AVX2)}",
        "qmatmul_q4_0_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_SSE)}",
        "qmatmul_q4_0_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_SSE)}",
        "qmatmul_q4_0_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Serial_Scalar)}",
        "qmatmul_q4_0_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_0_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ4_0(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ4_1, true, null,
        "q4_1_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_Scalar)}",
        "q4_1_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_AVX2)}",
        "q4_1_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_SSE)}",
        "q4_1_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ4_1_Scalar)}")]
    public abstract unsafe float VecDotQ4_1(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ4_1, true, null,
        "qmatmul_q4_1_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_AVX2)}",
        "qmatmul_q4_1_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_AVX2)}",
        "qmatmul_q4_1_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_AVX2)}",
        "qmatmul_q4_1_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_AVX2)}",
        "qmatmul_q4_1_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_SSE)}",
        "qmatmul_q4_1_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_SSE)}",
        "qmatmul_q4_1_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Serial_Scalar)}",
        "qmatmul_q4_1_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_1_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ4_1(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ5_0, true, null,
        "q5_0_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_FMA)}",
        "q5_0_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_AVX2)}",
        "q5_0_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_Scalar)}",
        "q5_0_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ5_0_Scalar)}")]
    public abstract unsafe float VecDotQ5_0(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ5_1, true, null,
        "q5_1_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_FMA)}",
        "q5_1_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_AVX2)}",
        "q5_1_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_Scalar)}",
        "q5_1_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ5_1_Scalar)}")]
    public abstract unsafe float VecDotQ5_1(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ4_NL,
        "q4_nl_fma",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_NL_AVX2)}",
        "q4_nl_avx2",  $"{NS}.{nameof(QuantizationKernels.VecDotQ4_NL_AVX2)}",
        "q4_nl_sse",   $"{NS}.{nameof(QuantizationKernels.VecDotQ4_NL_Scalar)}",
        "q4_nl_scalar",$"{NS}.{nameof(QuantizationKernels.VecDotQ4_NL_Scalar)}")]
    public abstract unsafe float VecDotQ4_NL(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ8_1, true, null,
        "q8_1_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_1_FMA)}",
        "q8_1_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ8_1_AVX2)}",
        "q8_1_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8_1_SSE)}",
        "q8_1_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ8_1_Scalar)}")]
    public abstract unsafe float VecDotQ8_1(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ2K, true, null,
        "q2k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ2K_FMA)}",
        "q2k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ2K_AVX2)}",
        "q2k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ2K_Scalar)}",
        "q2k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ2K_Scalar)}")]
    public abstract unsafe float VecDotQ2K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotQ8K, true, null,
        "q8k_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8K_FMA)}",
        "q8k_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotQ8K_AVX2)}",
        "q8k_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotQ8K_SSE)}",
        "q8k_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotQ8K_Scalar)}")]
    public abstract unsafe float VecDotQ8K(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotF32, true, null,
        "f32_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotF32_Scalar)}",
        "f32_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotF32_Scalar)}",
        "f32_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotF32_Scalar)}",
        "f32_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotF32_Scalar)}")]
    public abstract unsafe float VecDotF32(float* input, byte* rawWeights, int col, int inFeatures);

    [PuzzleCornerPiece(QuantizationKeys.KeyVecDotF16, true, null,
        "f16_fma",    $"{NS}.{nameof(QuantizationKernels.VecDotF16_Scalar)}",
        "f16_avx2",   $"{NS}.{nameof(QuantizationKernels.VecDotF16_Scalar)}",
        "f16_sse",    $"{NS}.{nameof(QuantizationKernels.VecDotF16_Scalar)}",
        "f16_scalar", $"{NS}.{nameof(QuantizationKernels.VecDotF16_Scalar)}")]
    public abstract unsafe float VecDotF16(float* input, byte* rawWeights, int col, int inFeatures);


    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ2K, true, null,
        "qmatmul_q2k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_FMA)}",
        "qmatmul_q2k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_FMA)}",
        "qmatmul_q2k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_AVX2)}",
        "qmatmul_q2k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_AVX2)}",
        "qmatmul_q2k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar)}",
        "qmatmul_q2k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar)}",
        "qmatmul_q2k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Serial_Scalar)}",
        "qmatmul_q2k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ2K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ2K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ3K, true, null,
        "qmatmul_q3k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_FMA)}",
        "qmatmul_q3k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_FMA)}",
        "qmatmul_q3k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_AVX2)}",
        "qmatmul_q3k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_AVX2)}",
        "qmatmul_q3k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar)}",
        "qmatmul_q3k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar)}",
        "qmatmul_q3k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Serial_Scalar)}",
        "qmatmul_q3k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ3K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ3K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ4K, true, null,
        "qmatmul_q4k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_FMA)}",
        "qmatmul_q4k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_FMA)}",
        "qmatmul_q4k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_AVX2)}",
        "qmatmul_q4k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_AVX2)}",
        "qmatmul_q4k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar)}",
        "qmatmul_q4k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar)}",
        "qmatmul_q4k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar)}",
        "qmatmul_q4k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ4K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ5K, true, null,
        "qmatmul_q5k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_FMA)}",
        "qmatmul_q5k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_FMA)}",
        "qmatmul_q5k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_AVX2)}",
        "qmatmul_q5k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_AVX2)}",
        "qmatmul_q5k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar)}",
        "qmatmul_q5k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar)}",
        "qmatmul_q5k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Serial_Scalar)}",
        "qmatmul_q5k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ5K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ8K, true, null,
        "qmatmul_q8k_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_FMA)}",
        "qmatmul_q8k_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_FMA)}",
        "qmatmul_q8k_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_AVX2)}",
        "qmatmul_q8k_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_AVX2)}",
        "qmatmul_q8k_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_SSE)}",
        "qmatmul_q8k_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_SSE)}",
        "qmatmul_q8k_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Serial_Scalar)}",
        "qmatmul_q8k_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8K_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ8K(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ8_1, true, null,
        "qmatmul_q8_1_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_FMA)}",
        "qmatmul_q8_1_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_FMA)}",
        "qmatmul_q8_1_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_AVX2)}",
        "qmatmul_q8_1_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_AVX2)}",
        "qmatmul_q8_1_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_SSE)}",
        "qmatmul_q8_1_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_SSE)}",
        "qmatmul_q8_1_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Serial_Scalar)}",
        "qmatmul_q8_1_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ8_1_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ8_1(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ5_1, true, null,
        "qmatmul_q5_1_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_FMA)}",
        "qmatmul_q5_1_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_FMA)}",
        "qmatmul_q5_1_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_AVX2)}",
        "qmatmul_q5_1_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_AVX2)}",
        "qmatmul_q5_1_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "qmatmul_q5_1_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}",
        "qmatmul_q5_1_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Serial_Scalar)}",
        "qmatmul_q5_1_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ5_1_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ5_1(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulQ4_NL, true, null,
        "qmatmul_q4_nl_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_AVX2)}",
        "qmatmul_q4_nl_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_AVX2)}",
        "qmatmul_q4_nl_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_AVX2)}",
        "qmatmul_q4_nl_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_AVX2)}",
        "qmatmul_q4_nl_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "qmatmul_q4_nl_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}",
        "qmatmul_q4_nl_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Serial_Scalar)}",
        "qmatmul_q4_nl_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulQ4_NL_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulQ4_NL(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulF32, true, null,
        "qmatmul_f32_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "qmatmul_f32_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "qmatmul_f32_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "qmatmul_f32_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "qmatmul_f32_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "qmatmul_f32_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}",
        "qmatmul_f32_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Serial_Scalar)}",
        "qmatmul_f32_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF32_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulF32(float* input, byte* rawWeights, float* output, int M, int K, int N);

    [PuzzleCornerPiece(QuantizationKeys.KeyQuantizedMatMulF16, true, null,
        "qmatmul_f16_serial_fma",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "qmatmul_f16_parallel_fma",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "qmatmul_f16_serial_avx2",   $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "qmatmul_f16_parallel_avx2", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "qmatmul_f16_serial_sse",    $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "qmatmul_f16_parallel_sse",  $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}",
        "qmatmul_f16_serial_scalar", $"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Serial_Scalar)}",
        "qmatmul_f16_parallel_scalar",$"{NS}.{nameof(QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar)}")]
    public abstract unsafe void QuantizedMatMulF16(float* input, byte* rawWeights, float* output, int M, int K, int N);



    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ8_0, true, null,
        "read_q8_0_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ8_0_Scalar)}",
        "read_q8_0_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ8_0_Scalar)}",
        "read_q8_0_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ8_0_Scalar)}",
        "read_q8_0_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ8_0_Scalar)}")]
    public abstract void ReadQ8_0(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ4_0, true, null,
        "read_q4_0_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_0_Scalar)}",
        "read_q4_0_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ4_0_Scalar)}",
        "read_q4_0_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_0_Scalar)}",
        "read_q4_0_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ4_0_Scalar)}")]
    public abstract void ReadQ4_0(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ4_1, true, null,
        "read_q4_1_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_1_Scalar)}",
        "read_q4_1_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ4_1_Scalar)}",
        "read_q4_1_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_1_Scalar)}",
        "read_q4_1_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ4_1_Scalar)}")]
    public abstract void ReadQ4_1(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ5_0, true, null,
        "read_q5_0_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_0_Scalar)}",
        "read_q5_0_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ5_0_Scalar)}",
        "read_q5_0_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_0_Scalar)}",
        "read_q5_0_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ5_0_Scalar)}")]
    public abstract void ReadQ5_0(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ5_1, true, null,
        "read_q5_1_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_1_Scalar)}",
        "read_q5_1_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ5_1_Scalar)}",
        "read_q5_1_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_1_Scalar)}",
        "read_q5_1_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ5_1_Scalar)}")]
    public abstract void ReadQ5_1(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ8_1, true, null,
        "read_q8_1_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ8_1_Scalar)}",
        "read_q8_1_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ8_1_Scalar)}",
        "read_q8_1_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ8_1_Scalar)}",
        "read_q8_1_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ8_1_Scalar)}")]
    public abstract void ReadQ8_1(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ4_NL, true, null,
        "read_q4_nl_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_NL_Scalar)}",
        "read_q4_nl_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ4_NL_Scalar)}",
        "read_q4_nl_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ4_NL_Scalar)}",
        "read_q4_nl_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ4_NL_Scalar)}")]
    public abstract void ReadQ4_NL(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ2K, true, null,
        "read_q2k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ2K_Scalar)}",
        "read_q2k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ2K_Scalar)}",
        "read_q2k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ2K_Scalar)}",
        "read_q2k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ2K_Scalar)}")]
    public abstract void ReadQ2K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ3K, true, null,
        "read_q3k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ3_K_Scalar)}",
        "read_q3k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ3_K_Scalar)}",
        "read_q3k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ3_K_Scalar)}",
        "read_q3k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ3_K_Scalar)}")]
    public abstract void ReadQ3K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ4K, true, null,
        "read_q4k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ4K_Scalar)}",
        "read_q4k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ4K_Scalar)}",
        "read_q4k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ4K_Scalar)}",
        "read_q4k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ4K_Scalar)}")]
    public abstract void ReadQ4K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ5K, true, null,
        "read_q5k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_K_Scalar)}",
        "read_q5k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ5_K_Scalar)}",
        "read_q5k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ5_K_Scalar)}",
        "read_q5k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ5_K_Scalar)}")]
    public abstract void ReadQ5K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ6K, true, null,
        "read_q6k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ6K_Scalar)}",
        "read_q6k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ6K_Scalar)}",
        "read_q6k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ6K_Scalar)}",
        "read_q6k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ6K_Scalar)}")]
    public abstract void ReadQ6K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadQ8K, true, null,
        "read_q8k_fma",    $"{NS}.{nameof(QuantizationKernels.ReadQ8K_Scalar)}",
        "read_q8k_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadQ8K_Scalar)}",
        "read_q8k_sse",    $"{NS}.{nameof(QuantizationKernels.ReadQ8K_Scalar)}",
        "read_q8k_scalar", $"{NS}.{nameof(QuantizationKernels.ReadQ8K_Scalar)}")]
    public abstract void ReadQ8K(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadF32, true, null,
        "read_f32_fma",    $"{NS}.{nameof(QuantizationKernels.ReadF32_Scalar)}",
        "read_f32_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadF32_Scalar)}",
        "read_f32_sse",    $"{NS}.{nameof(QuantizationKernels.ReadF32_Scalar)}",
        "read_f32_scalar", $"{NS}.{nameof(QuantizationKernels.ReadF32_Scalar)}")]
    public abstract void ReadF32(BinaryReader reader, Span<float> data, int n);

    [PuzzleCornerPiece(QuantizationKeys.KeyReadF16, true, null,
        "read_f16_fma",    $"{NS}.{nameof(QuantizationKernels.ReadF16_F16C)}",
        "read_f16_avx2",   $"{NS}.{nameof(QuantizationKernels.ReadF16_F16C)}",
        "read_f16_sse",    $"{NS}.{nameof(QuantizationKernels.ReadF16_Scalar)}",
        "read_f16_scalar", $"{NS}.{nameof(QuantizationKernels.ReadF16_Scalar)}")]
    public abstract void ReadF16(BinaryReader reader, Span<float> data, int n);


    [PuzzleCornerPiece(QuantizationKeys.KeyHSum256, true, null,
        "hsum_fma",   $"{MH}.{nameof(MathHelpers.HSum256_Avx)}",
        "hsum_avx2",  $"{MH}.{nameof(MathHelpers.HSum256_Avx)}",
        "hsum_sse3",  $"{MH}.{nameof(MathHelpers.HSum256_Sse3)}",
        "hsum_sse",   $"{MH}.{nameof(MathHelpers.HSum256_Sse3)}",
        "hsum_scalar", $"{MH}.{nameof(MathHelpers.HSum256_Scalar)}")]
    public abstract float HSum256(Vector256<float> v);

    [PuzzleCornerPiece(QuantizationKeys.KeyHalfToFloat, true, null,
        "halftofloat_fma",   $"{NS}.{nameof(QuantizationKernels.HalfToFloat_F16C)}",
        "halftofloat_avx2",  $"{NS}.{nameof(QuantizationKernels.HalfToFloat_F16C)}",
        "halftofloat_sse",   $"{NS}.{nameof(QuantizationKernels.HalfToFloat_Scalar)}",
        "halftofloat_scalar", $"{NS}.{nameof(QuantizationKernels.HalfToFloat_Scalar)}")]
    public abstract float HalfToFloat(ushort half);

    [PuzzleCornerPiece(QuantizationKeys.KeyFloatToHalf, true, null,
        "floattohalf_fma",   $"{NS}.{nameof(QuantizationKernels.FloatToHalf_F16C)}",
        "floattohalf_avx2",  $"{NS}.{nameof(QuantizationKernels.FloatToHalf_F16C)}",
        "floattohalf_sse",   $"{NS}.{nameof(QuantizationKernels.FloatToHalf_Scalar)}",
        "floattohalf_scalar", $"{NS}.{nameof(QuantizationKernels.FloatToHalf_Scalar)}")]
    public abstract ushort FloatToHalf(float f);

    [PuzzleCornerPiece(QuantizationKeys.KeyGetScaleMinK4_Scale, true, null,
        "getscalemink4_scale_fma",    $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Scale_Scalar)}",
        "getscalemink4_scale_avx2",   $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Scale_Scalar)}",
        "getscalemink4_scale_sse",    $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Scale_Scalar)}",
        "getscalemink4_scale_scalar", $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Scale_Scalar)}")]
    public abstract unsafe byte GetScaleMinK4_Scale(int j, byte* scales);

    [PuzzleCornerPiece(QuantizationKeys.KeyGetScaleMinK4_Min, true, null,
        "getscalemink4_min_fma",    $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Min_Scalar)}",
        "getscalemink4_min_avx2",   $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Min_Scalar)}",
        "getscalemink4_min_sse",    $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Min_Scalar)}",
        "getscalemink4_min_scalar", $"{NS}.{nameof(QuantizationKernels.GetScaleMinK4_Min_Scalar)}")]
    public abstract unsafe byte GetScaleMinK4_Min(int j, byte* scales);
}
