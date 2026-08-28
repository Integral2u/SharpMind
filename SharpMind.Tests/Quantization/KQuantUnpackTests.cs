using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// The Q4_K, Q6_K and Q5_0 vector kernels unpack packed weights straight from
/// memory with vector shifts and masks. Q4_K reads a 32-value sub-block as one
/// contiguous nibble run, Q6_K reassembles four interleaved groups from two
/// nibble bytes and one high-bit byte, and Q5_0 pulls each weight's fifth bit
/// out of a 32-bit mask with a per-lane shift. Each is easy to get subtly wrong
/// — the wrong nibble half, the wrong bit pair, a mask applied before a shift —
/// and the results would still look like plausible weights.
///
/// The existing agreement test only uses K as a multiple of the super-block, so
/// it never sees a column that starts mid-block (K=896 puts every odd column at
/// offset 128, which is exactly what qwen2-0.5b's ffn_down does), a column that
/// starts off a 32 boundary (falls back to the per-weight path, which must agree
/// too), or a partial final block. These do.
/// </summary>
public class KQuantUnpackTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var dtype in new[] { QuantDType.Q4_K, QuantDType.Q6_K, QuantDType.Q5_0 })
            // 256/512: whole super-blocks. 896, 4864: qwen2 shapes, odd columns
            // start at offset 128. 288/320/384: offsets 32/64/128. 8/20/137:
            // partial blocks and columns off any alignment.
            foreach (int k in new[] { 8, 20, 137, 256, 288, 320, 384, 512, 896, 4864 })
                yield return new object[] { dtype, k };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public unsafe void VectorisedTiersAgreeWithScalar(QuantDType dtype, int inFeatures)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        const int nCols = 3;
        const int Poison = 64;

        int totalBytes = (int)QuantizationOps.GetRawTensorByteCount([inFeatures, nCols], dtype);
        var raw = new byte[totalBytes];
        var rng = new Random(4321 + inFeatures);
        rng.NextBytes(raw);

        // Pin every fp16 super-scale so random bytes cannot make it NaN/Inf.
        // Q4_K: d at 0, dmin at 2 of 144. Q6_K: d at 208 of 210. Q5_0: d at 0 of 22.
        (int blockBytes, int dOff, int minOff) = dtype switch
        {
            QuantDType.Q4_K => (144, 0, 2),
            QuantDType.Q6_K => (210, 208, -1),
            _ => (22, 0, -1),
        };
        for (int off = 0; off + blockBytes <= raw.Length; off += blockBytes)
        {
            raw[off + dOff] = 0x00; raw[off + dOff + 1] = 0x38;                       // 0.5
            if (minOff >= 0) { raw[off + minOff] = 0x00; raw[off + minOff + 1] = 0x34; } // 0.25
        }

        // Loud padding past K: an unpack that reaches into the next block's
        // input picks up 1e6 and blows the tolerance rather than reading zeros.
        var input = new float[inFeatures + Poison];
        for (int i = 0; i < inFeatures; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = inFeatures; i < input.Length; i++) input[i] = 1e6f;

        var scalar = QuantizationFactory.Create(HardwareTier.Scalar);
        var avx2 = QuantizationFactory.Create(HardwareTier.AVX2);
        var fma = QuantizationFactory.Create(HardwareTier.FMA);

        fixed (float* pIn = input)
        fixed (byte* pW = raw)
        {
            for (int c = 0; c < nCols; c++)
            {
                float expected = Dot(scalar, dtype, pIn, pW, c, inFeatures);
                float gotAvx2 = Dot(avx2, dtype, pIn, pW, c, inFeatures);
                float gotFma = Dot(fma, dtype, pIn, pW, c, inFeatures);

                // Scalar accumulates in double; the vector kernels in float
                // across up to 4864 terms, so the bound scales with magnitude.
                float tol = 2e-4f * Math.Max(1f, Math.Abs(expected));
                Assert.True(Math.Abs(expected - gotAvx2) <= tol,
                    $"{dtype} K={inFeatures} col={c}: scalar={expected}, avx2={gotAvx2}");
                Assert.True(Math.Abs(expected - gotFma) <= tol,
                    $"{dtype} K={inFeatures} col={c}: scalar={expected}, fma={gotFma}");
            }
        }
    }

    private static unsafe float Dot(QuantizationOps ops, QuantDType dtype, float* pIn, byte* pW, int col, int k) => dtype switch
    {
        QuantDType.Q4_K => ops.VecDotQ4K(pIn, pW, col, k),
        QuantDType.Q6_K => ops.VecDotQ6K(pIn, pW, col, k),
        _ => ops.VecDotQ5_0(pIn, pW, col, k),
    };
}
