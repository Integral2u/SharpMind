using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// The quantized matmuls' M &gt; 1 path now dequantizes each weight column once
/// into an L2-resident scratch tile and runs the blocked four-row microkernel
/// against it (<c>QuantBlockedColumns</c>), instead of re-unpacking every column
/// once per row. These tests pit the new Serial/Parallel FMA path against the
/// scalar per-row reference (which accumulates in double) for every rewired
/// format, at shapes that hit each seam: rows past the last full block of four,
/// row counts past one 16-row tile, column counts that cross a scratch-tile
/// boundary, K-quant columns that start mid-super-block (K=896 puts odd columns
/// at offset 128), columns off any 32 alignment, and partial final blocks.
/// Outputs are NaN-poisoned first so an unwritten slot fails loudly, and the
/// M &lt;= 1 decode path is asserted unchanged.
/// </summary>
public class QuantPrefillBlockingTests
{
    public static IEnumerable<object[]> Cases()
    {
        // K-quants are tested only at K values real GGUFs can produce (rows a
        // multiple of 256 — columns start at super-block offset 0 or, for
        // K=896, 128): at other K the per-row VecDot reference itself skips a
        // column's tail block and over-reads the input on partial halves
        // (consistently between its scalar and vector forms, so its own tests
        // pass), and the blocked path's correct handling would "fail" against
        // it. Q5_0/Q8_0 blocks are per-column, so any K is fair game there.
        foreach (var dtype in new[] { QuantDType.Q4_K, QuantDType.Q6_K })
            foreach (int k in new[] { 256, 896 })
                foreach (int m in new[] { 1, 2, 5, 16, 33 })
                    yield return new object[] { dtype, k, m };
        foreach (var dtype in new[] { QuantDType.Q5_0, QuantDType.Q8_0 })
            foreach (int k in new[] { 137, 256, 320, 896 })
                foreach (int m in new[] { 1, 2, 5, 16, 33 })
                    yield return new object[] { dtype, k, m };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public unsafe void BlockedPrefillAgreesWithScalarReference(QuantDType dtype, int inFeatures, int M)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        // 80 columns crosses one scratch-tile boundary at K=896 (tile = 72 cols).
        const int nCols = 80;

        int totalBytes = (int)QuantizationOps.GetRawTensorByteCount([inFeatures, nCols], dtype);
        var raw = new byte[totalBytes];
        var rng = new Random(9182 + inFeatures * 31 + M);
        rng.NextBytes(raw);

        // Pin every fp16 scale so random bytes cannot make it NaN/Inf.
        (int blockBytes, int dOff, int minOff) = dtype switch
        {
            QuantDType.Q4_K => (144, 0, 2),
            QuantDType.Q6_K => (210, 208, -1),
            QuantDType.Q5_0 => (22, 0, -1),
            _ => (34, 0, -1),
        };
        for (int off = 0; off + blockBytes <= raw.Length; off += blockBytes)
        {
            raw[off + dOff] = 0x00; raw[off + dOff + 1] = 0x38;                       // 0.5
            if (minOff >= 0) { raw[off + minOff] = 0x00; raw[off + minOff + 1] = 0x34; } // 0.25
        }

        var input = new float[(long)M * inFeatures];
        for (int i = 0; i < input.Length; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);

        var expected = new float[(long)M * nCols];
        var serial = new float[(long)M * nCols];
        var parallel = new float[(long)M * nCols];
        Array.Fill(serial, float.NaN);
        Array.Fill(parallel, float.NaN);

        fixed (float* pIn = input)
        fixed (byte* pW = raw)
        fixed (float* pE = expected)
        fixed (float* pS = serial)
        fixed (float* pP = parallel)
        {
            RunScalar(dtype, pIn, pW, pE, M, inFeatures, nCols);
            RunFma(dtype, serialVariant: true, pIn, pW, pS, M, inFeatures, nCols);
            RunFma(dtype, serialVariant: false, pIn, pW, pP, M, inFeatures, nCols);
        }

        for (long i = 0; i < expected.Length; i++)
        {
            // Scalar accumulates in double; the blocked kernel in float, in a
            // different order, so the bound scales with magnitude — plus a small
            // absolute floor for near-cancelling sums, whose error scales with
            // the accumulated magnitude rather than the tiny result.
            float tol = 2e-4f * Math.Max(1f, Math.Abs(expected[i])) + 0.05f;
            Assert.True(Math.Abs(expected[i] - serial[i]) <= tol,
                $"{dtype} K={inFeatures} M={M} serial[{i / nCols},{i % nCols}]: scalar={expected[i]}, blocked={serial[i]}");
            Assert.True(Math.Abs(expected[i] - parallel[i]) <= tol,
                $"{dtype} K={inFeatures} M={M} parallel[{i / nCols},{i % nCols}]: scalar={expected[i]}, blocked={parallel[i]}");
        }
    }

    private static unsafe void RunScalar(QuantDType dtype, float* i, byte* w, float* o, int M, int K, int N)
    {
        switch (dtype)
        {
            case QuantDType.Q4_K: QuantizationKernels.QuantizedMatMulQ4K_Serial_Scalar(i, w, o, M, K, N); break;
            case QuantDType.Q6_K: QuantizationKernels.QuantizedMatMulQ6K_Serial_Scalar(i, w, o, M, K, N); break;
            case QuantDType.Q5_0: QuantizationKernels.QuantizedMatMulQ5_0_Serial_Scalar(i, w, o, M, K, N); break;
            default: QuantizationKernels.QuantizedMatMulQ8_0_Serial_Scalar(i, w, o, M, K, N); break;
        }
    }

    private static unsafe void RunFma(QuantDType dtype, bool serialVariant, float* i, byte* w, float* o, int M, int K, int N)
    {
        switch (dtype, serialVariant)
        {
            case (QuantDType.Q4_K, true): QuantizationKernels.QuantizedMatMulQ4K_Serial_FMA(i, w, o, M, K, N); break;
            case (QuantDType.Q4_K, false): QuantizationKernels.QuantizedMatMulQ4K_Parallel_FMA(i, w, o, M, K, N); break;
            case (QuantDType.Q6_K, true): QuantizationKernels.QuantizedMatMulQ6K_Serial_FMA(i, w, o, M, K, N); break;
            case (QuantDType.Q6_K, false): QuantizationKernels.QuantizedMatMulQ6K_Parallel_FMA(i, w, o, M, K, N); break;
            case (QuantDType.Q5_0, true): QuantizationKernels.QuantizedMatMulQ5_0_Serial_FMA(i, w, o, M, K, N); break;
            case (QuantDType.Q5_0, false): QuantizationKernels.QuantizedMatMulQ5_0_Parallel_FMA(i, w, o, M, K, N); break;
            case (QuantDType.Q8_0, true): QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA(i, w, o, M, K, N); break;
            default: QuantizationKernels.QuantizedMatMulQ8_0_Parallel_FMA(i, w, o, M, K, N); break;
        }
    }
}
