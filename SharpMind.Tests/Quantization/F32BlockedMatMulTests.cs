using System.Runtime.Intrinsics.X86;
using SharpMind.Core.Quantization;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// The F32 matmul's M &gt; 1 path now processes four rows per weight vector
/// (<c>F32BlockedColumns</c>, the F32 twin of the F16 kernel from the prefill
/// work). Same edges as the F16 test: rows past the last full block of four, a
/// K tail that is not a multiple of eight, and the column split of the parallel
/// variant. Reference is computed in double here, with the tolerance scaled to
/// the accumulated magnitude, for the reasons the F16 test spells out.
/// </summary>
public sealed class F32BlockedMatMulTests
{
    [Theory]
    [InlineData(1, 896, 64)]     // decode: no blocking at all
    [InlineData(1, 896, 4864)]   // decode, chunked across every core
    [InlineData(2, 896, 64)]     // shorter than one row block
    [InlineData(4, 896, 128)]    // exactly one row block
    [InlineData(7, 896, 96)]     // one block + 3 tail rows
    [InlineData(16, 896, 256)]   // exactly one row tile
    [InlineData(37, 896, 256)]   // two tiles + a 5-row tail tile with a 1-row remainder
    [InlineData(256, 896, 1024)] // a training step's row count, column chunking engaged
    [InlineData(13, 133, 71)]    // awkward everywhere: rows, K tail, odd column count
    public unsafe void QuantizedMatMulF32_FMA_AgreesWithDoubleReference_ForAnyRowCount(int M, int K, int N)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        var rng = new Random(4242);
        var weights = new float[(long)K * N];
        for (int i = 0; i < weights.Length; i++) weights[i] = (float)(rng.NextDouble() * 0.4 - 0.2);
        var input = new float[(long)M * K];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var serial = new float[(long)M * N];
        var parallel = new float[(long)M * N];
        // Poison so an unwritten output slot fails loudly instead of reading as 0.
        Array.Fill(serial, float.NaN);
        Array.Fill(parallel, float.NaN);

        fixed (float* pW = weights)
        fixed (float* pIn = input)
        fixed (float* pS = serial)
        fixed (float* pP = parallel)
        {
            QuantizationKernels.QuantizedMatMulF32_Serial_FMA(pIn, (byte*)pW, pS, M, K, N);
            QuantizationKernels.QuantizedMatMulF32_Parallel_FMA(pIn, (byte*)pW, pP, M, K, N);
        }

        for (int m = 0; m < M; m++)
        {
            for (int n = 0; n < N; n++)
            {
                double acc = 0, absAcc = 0;
                for (int k = 0; k < K; k++)
                {
                    double term = (double)input[(long)m * K + k] * (double)weights[(long)n * K + k];
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
