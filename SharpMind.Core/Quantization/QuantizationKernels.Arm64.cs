using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace SharpMind.Core.Quantization;

public static partial class QuantizationKernels
{
    /// <summary>
    /// Q8_0 vec-dot on Arm64 NEON, the 128-bit counterpart of
    /// <see cref="VecDotQ8_0_AVX2"/>: eight int8 weights are widened to two
    /// <see cref="Vector128{T}"/> of float and fused into two four-lane
    /// accumulators, so a full 32-value block is four widening loads and eight
    /// multiply-adds. The block scale is applied to the widened weights rather
    /// than to the input, so <c>d</c> is broadcast once per block.
    /// </summary>
    /// <remarks>
    /// A block's lanes accumulate in a different order from the scalar kernel, so
    /// results agree to float precision rather than bit-for-bit; every caller
    /// compares with a tolerance. The tail is scalar because the last block of a
    /// row can be shorter than 32 when <paramref name="inFeatures"/> is not a
    /// multiple of QK.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe float VecDotQ8_0_Arm64(float* input, byte* rawWeights, int col, int inFeatures)
    {
        const int BLOCK_BYTES = 34;
        int nBlocks = (inFeatures + QK - 1) / QK;
        double sum = 0;

        for (int b = 0; b < nBlocks; b++)
        {
            byte* block = rawWeights + (long)col * nBlocks * BLOCK_BYTES + b * BLOCK_BYTES;
            float d = HalfToFloat_Scalar(*(ushort*)block);
            sbyte* values = (sbyte*)(block + 2);
            int blockEnd = Math.Min(QK, inFeatures - b * QK);
            float* pIn = input + b * QK;

            var vd = Vector128.Create(d);
            var vacc0 = Vector128<float>.Zero;
            var vacc1 = Vector128<float>.Zero;

            int i = 0;
            for (; i <= blockEnd - 8; i += 8)
            {
                (Vector128<float> lo, Vector128<float> hi) = WidenQ8_8ToFloat(values + i);
                vacc0 = AdvSimd.Add(vacc0,
                    AdvSimd.Multiply(AdvSimd.LoadVector128(pIn + i), AdvSimd.Multiply(lo, vd)));
                vacc1 = AdvSimd.Add(vacc1,
                    AdvSimd.Multiply(AdvSimd.LoadVector128(pIn + i + 4), AdvSimd.Multiply(hi, vd)));
            }

            sum += Vector128.Sum(AdvSimd.Add(vacc0, vacc1));

            for (; i < blockEnd; i++)
                sum += pIn[i] * (values[i] * d);
        }

        return (float)sum;
    }

    /// <summary>
    /// Eight packed int8 values as two float vectors (low four, high four).
    /// sbyte -> short -> int -> float is two SXTL pairs and two SCVTF, which is
    /// cheaper than going through int16 arithmetic: the product has to end up in
    /// float anyway because the input is float.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe (Vector128<float> Lo, Vector128<float> Hi) WidenQ8_8ToFloat(sbyte* values)
    {
        Vector128<short> shorts = AdvSimd.SignExtendWideningLower(AdvSimd.LoadVector64(values));
        Vector128<int> lo = AdvSimd.SignExtendWideningLower(shorts.GetLower());
        Vector128<int> hi = AdvSimd.SignExtendWideningUpper(shorts);
        return (AdvSimd.ConvertToSingle(lo), AdvSimd.ConvertToSingle(hi));
    }

    /// <summary>
    /// Serial Q8_0 matmul. Rows run in order and each row is one vec-dot per
    /// column; there is no blocked microkernel yet, so a multi-row (prefill) call
    /// costs the same per row as decode. That is the next NEON step, not a
    /// correctness gap.
    /// </summary>
    public static unsafe void QuantizedMatMulQ8_0_Serial_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        for (int row = 0; row < M; row++)
        {
            float* pInRow = input + (long)row * K;
            float* pOutRow = output + (long)row * N;
            for (int col = 0; col < N; col++)
                pOutRow[col] = VecDotQ8_0_Arm64(pInRow, rawWeights, col, K);
        }
    }

    /// <summary>
    /// Parallel Q8_0 matmul: decode (M=1) splits the columns through
    /// <see cref="DecodeParallel"/>, exactly like the FMA and AVX2 variants, and
    /// prefill splits the rows. Column splitting is what keeps a single-token
    /// call on every core, which is the case that matters for chat latency.
    /// </summary>
    public static unsafe void QuantizedMatMulQ8_0_Parallel_Arm64(
        float* input, byte* rawWeights, float* output,
        int M, int K, int N)
    {
        if (M <= 1)
        {
            DecodeParallel(VecDotQ8_0_Arm64, input, rawWeights, output, K, N);
        }
        else
        {
            Parallel.For(0, M, row =>
            {
                float* pInRow = input + (long)row * K;
                float* pOutRow = output + (long)row * N;
                for (int col = 0; col < N; col++)
                    pOutRow[col] = VecDotQ8_0_Arm64(pInRow, rawWeights, col, K);
            });
        }
    }
}
