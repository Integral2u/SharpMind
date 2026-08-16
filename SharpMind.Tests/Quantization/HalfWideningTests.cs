using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Quantization;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// Guards <see cref="QuantizationKernels.WidenHalf8"/> and the F16 vecdot built on
/// it. The widening is arithmetic rather than a hardware convert, so the exactness
/// sweep below is what makes it safe to use on real model weights.
/// </summary>
public class HalfWideningTests
{
    [Fact]
    public unsafe void WidenHalf8_MatchesHalfToFloat_ForEveryBitPattern()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        var patterns = new ushort[65536];
        for (int i = 0; i < patterns.Length; i++) patterns[i] = (ushort)i;

        var got = new float[8];
        fixed (ushort* p = patterns)
        fixed (float* g = got)
        {
            for (int i = 0; i < patterns.Length; i += 8)
            {
                Vector256.Store(QuantizationKernels.WidenHalf8(p + i), g);
                for (int j = 0; j < 8; j++)
                {
                    ushort bits = (ushort)(i + j);
                    float expected = (float)BitConverter.UInt16BitsToHalf(bits);
                    float actual = got[j];

                    if (float.IsNaN(expected))
                    {
                        Assert.True(float.IsNaN(actual), $"half 0x{bits:X4} should widen to NaN, got {actual}");
                        continue;
                    }
                    // Bit comparison, not ==, so -0.0 and +0.0 are held apart.
                    Assert.True(
                        BitConverter.SingleToInt32Bits(expected) == BitConverter.SingleToInt32Bits(actual),
                        $"half 0x{bits:X4}: expected {expected} (0x{BitConverter.SingleToInt32Bits(expected):X8}), " +
                        $"got {actual} (0x{BitConverter.SingleToInt32Bits(actual):X8})");
                }
            }
        }
    }

    /// <summary>
    /// Covers the vecdot's loop structure — the 32-wide body, the 8-wide body and
    /// the scalar tail — at lengths that exercise each in turn, against the scalar
    /// kernel. Lengths are deliberately not all multiples of 32.
    /// </summary>
    [Theory]
    [InlineData(7)]    // tail only
    [InlineData(8)]    // one 8-wide step
    [InlineData(31)]   // 8-wide steps + tail, never reaches the 32-wide body
    [InlineData(32)]   // exactly one 32-wide step
    [InlineData(96)]   // three 32-wide steps
    [InlineData(133)]  // 32-wide + 8-wide + tail
    [InlineData(896)]  // Qwen2-0.5B hidden size
    public unsafe void VecDotF16_FMA_AgreesWithScalar(int inFeatures)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        var rng = new Random(20260816);
        var weights = new ushort[inFeatures];
        var input = new float[inFeatures];
        for (int i = 0; i < inFeatures; i++)
        {
            weights[i] = QuantizationKernels.FloatToHalf_Scalar((float)(rng.NextDouble() * 0.4 - 0.2));
            input[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        // Salt in the awkward encodings the random draw will never produce:
        // +0, -0, the smallest subnormal, and the largest finite half.
        if (inFeatures >= 4)
        {
            weights[0] = 0x0000;
            weights[1] = 0x8000;
            weights[2] = 0x0001;
            weights[inFeatures - 1] = 0x7BFF;
        }

        fixed (ushort* pW = weights)
        fixed (float* pIn = input)
        {
            float scalar = QuantizationKernels.VecDotF16_Scalar(pIn, (byte*)pW, 0, inFeatures);
            float fma = QuantizationKernels.VecDotF16_FMA(pIn, (byte*)pW, 0, inFeatures);
            // Different summation orders, so compare on relative error, not bits.
            Assert.True(
                Math.Abs(scalar - fma) <= 1e-5f * Math.Max(1f, Math.Abs(scalar)),
                $"inFeatures={inFeatures}: scalar={scalar}, fma={fma}");
        }
    }

    /// <summary>
    /// The multi-column path: a matmul must place every column's dot product at the
    /// right output index, whichever kernel computes it. This is the check that the
    /// column-parallel chunking still lines up with the serial result.
    /// </summary>
    [Fact]
    public unsafe void QuantizedMatMulF16_ParallelFMA_AgreesWithScalarKernel()
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        const int K = 896, N = 1024;   // N > the 512-column floor, so chunking engages
        var rng = new Random(7);
        var weights = new ushort[(long)K * N];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = QuantizationKernels.FloatToHalf_Scalar((float)(rng.NextDouble() * 0.4 - 0.2));
        var input = new float[K];
        for (int i = 0; i < K; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var expected = new float[N];
        var actual = new float[N];
        fixed (ushort* pW = weights)
        fixed (float* pIn = input)
        fixed (float* pE = expected)
        fixed (float* pA = actual)
        {
            QuantizationKernels.QuantizedMatMulF16_Parallel_Scalar(pIn, (byte*)pW, pE, 1, K, N);
            QuantizationKernels.QuantizedMatMulF16_Parallel_FMA(pIn, (byte*)pW, pA, 1, K, N);
        }

        for (int i = 0; i < N; i++)
            Assert.True(
                Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                $"column {i}: scalar={expected[i]}, fma={actual[i]}");
    }

    /// <summary>
    /// The M &gt; 1 path processes four rows per widened weight vector, so it has two
    /// edges the single-row path does not: rows past the last full block of four,
    /// and the column split used by the parallel variant. Row counts here are
    /// deliberately not multiples of four, and the sizes straddle the point where
    /// the parallel kernel starts handing columns to more than one thread.
    /// </summary>
    [Theory]
    [InlineData(1, 896, 64)]     // decode: no blocking at all
    [InlineData(1, 896, 17)]     // decode, below the parallel-chunking threshold
    [InlineData(1, 896, 71)]     // decode, column count not a multiple of the chunk quantum
    [InlineData(1, 896, 4864)]   // decode, chunked across every core
    [InlineData(1, 4864, 896)]   // decode, down-projection shape
    [InlineData(2, 896, 64)]     // shorter than one row block
    [InlineData(4, 896, 128)]    // exactly one row block
    [InlineData(7, 896, 96)]     // one block + 3 tail rows
    [InlineData(128, 896, 256)]  // a full prefill chunk
    [InlineData(13, 133, 71)]    // awkward everywhere: rows, K tail, odd column count
    public unsafe void QuantizedMatMulF16_FMA_AgreesWithScalar_ForAnyRowCount(int M, int K, int N)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        var rng = new Random(31337);
        var weights = new ushort[(long)K * N];
        for (int i = 0; i < weights.Length; i++)
            weights[i] = QuantizationKernels.FloatToHalf_Scalar((float)(rng.NextDouble() * 0.4 - 0.2));
        var input = new float[(long)M * K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var serial = new float[(long)M * N];
        var parallel = new float[(long)M * N];

        fixed (ushort* pW = weights)
        fixed (float* pIn = input)
        fixed (float* pS = serial)
        fixed (float* pP = parallel)
        {
            QuantizationKernels.QuantizedMatMulF16_Serial_FMA(pIn, (byte*)pW, pS, M, K, N);
            QuantizationKernels.QuantizedMatMulF16_Parallel_FMA(pIn, (byte*)pW, pP, M, K, N);
        }

        // Reference in double, computed here rather than taken from another kernel.
        // QuantizedMatMulF16_Serial_Scalar accumulates in float, so over a 4864-long
        // K it carries more error than the kernels under test and makes a poor
        // oracle — it disagreed with both by 1.5e-5 relative before this changed.
        //
        // Tolerance scales with the accumulated magnitude, not the final value. A
        // dot product whose terms largely cancel has a small result and a large
        // error budget, and judging it against its own tiny output would demand a
        // precision float never had. A misplaced column still fails loudly: that
        // error is the size of the sum itself, not of its rounding.
        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                double acc = 0, absAcc = 0;
                for (int k = 0; k < K; k++)
                {
                    double term = (double)input[(long)m * K + k]
                                * (double)(float)BitConverter.UInt16BitsToHalf(weights[(long)n * K + k]);
                    acc += term;
                    absAcc += Math.Abs(term);
                }
                double tol = 1e-6 * Math.Max(1.0, absAcc);
                long i = (long)m * N + n;
                Assert.True(Math.Abs(acc - serial[i]) <= tol,
                    $"serial [{m},{n}]: exact={acc}, fma={serial[i]}, tol={tol}");
                Assert.True(Math.Abs(acc - parallel[i]) <= tol,
                    $"parallel [{m},{n}]: exact={acc}, fma={parallel[i]}, tol={tol}");
            }
        }
    }
}
