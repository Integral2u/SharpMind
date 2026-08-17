using System;
using System.Collections.Generic;
using System.Runtime.Intrinsics.X86;
using SharpMind.Core;
using SharpMind.Core.Quantization;
using Xunit;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// Q8_0 and Q8_1 pack 32 values per block, and their vectorised vecdots consume
/// a block with a 16-wide loop, then an 8-wide loop, then a scalar tail. The
/// loop bounds decide how much of each block reaches the vector path — they were
/// once set so that elements 24..31 of every full block fell through to the
/// scalar tail, which cost about a third of Q8_0's throughput without changing
/// any result.
///
/// The existing agreement tests only use K as a multiple of the block size, so
/// they never exercise a partial final block. These do: an off-by-one in those
/// bounds turns from a slowdown into a read past the end of the block exactly
/// there.
/// </summary>
public class Q8BlockTailTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var dtype in new[] { QuantDType.Q8_0, QuantDType.Q8_1 })
            // 32 = one exact block; the rest leave a partial block of every
            // interesting size, including shorter than each loop's stride.
            foreach (int k in new[] { 8, 15, 20, 32, 33, 40, 63, 64, 96, 137, 896 })
                yield return new object[] { dtype, k };
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public unsafe void VectorisedTiersAgreeWithScalar(QuantDType dtype, int inFeatures)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported) return;

        const int nCols = 3;
        const int Poison = 64;   // elements of deliberately loud padding

        int totalBytes = (int)QuantizationOps.GetRawTensorByteCount([inFeatures, nCols], dtype);
        var raw = new byte[totalBytes];
        var rng = new Random(1234 + inFeatures);
        rng.NextBytes(raw);

        // Block scales are fp16; random bytes can make them NaN or Inf, which
        // would compare unequal for reasons that have nothing to do with the
        // loop bounds. Pin every scale (and Q8_1's offset) to something tame.
        int blockBytes = dtype == QuantDType.Q8_0 ? 34 : 36;
        for (int off = 0; off + blockBytes <= raw.Length; off += blockBytes)
        {
            raw[off] = 0x00; raw[off + 1] = 0x38;          // 0.5
            if (dtype == QuantDType.Q8_1) { raw[off + 2] = 0x00; raw[off + 3] = 0xB8; }  // -0.5
        }

        // Reading past a partial block lands in whatever happens to follow the
        // input array, which is often zeroed and therefore silent. Give it a loud
        // neighbour instead: any overread picks up 1e6 and blows the tolerance
        // apart, so a bound that walks off the end fails here rather than
        // depending on the allocator's mood.
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
                float expected, gotAvx2, gotFma;
                if (dtype == QuantDType.Q8_0)
                {
                    expected = scalar.VecDotQ8_0(pIn, pW, c, inFeatures);
                    gotAvx2 = avx2.VecDotQ8_0(pIn, pW, c, inFeatures);
                    gotFma = fma.VecDotQ8_0(pIn, pW, c, inFeatures);
                }
                else
                {
                    expected = scalar.VecDotQ8_1(pIn, pW, c, inFeatures);
                    gotAvx2 = avx2.VecDotQ8_1(pIn, pW, c, inFeatures);
                    gotFma = fma.VecDotQ8_1(pIn, pW, c, inFeatures);
                }

                float tol = 1e-4f * Math.Max(1f, Math.Abs(expected));
                Assert.True(Math.Abs(expected - gotAvx2) <= tol,
                    $"{dtype} K={inFeatures} col={c}: scalar={expected}, avx2={gotAvx2}");
                Assert.True(Math.Abs(expected - gotFma) <= tol,
                    $"{dtype} K={inFeatures} col={c}: scalar={expected}, fma={gotFma}");
            }
        }
    }
}
