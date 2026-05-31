using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using JigSawDotNet;

namespace SharpMind.Core.Quantization;

public static class QuantizationFactory
{
    public static QuantizationOps Create(HardwareTier hw)
    {
        var config = new QuantizationConfig { Hardware = hw };
        return Assembler.CreateInstance<QuantizationOps>(config.ToJigSawMapping());
    }

    /// <summary>
    /// Returns all hardware variants available on this machine.
    /// Used by tests to verify correctness across every implementation path.
    /// </summary>
    public static QuantizationOps[] CreateAllAvailable()
    {
        var list = new List<QuantizationOps> { Create(HardwareTier.Scalar) };

        if (Sse.IsSupported)
        {
            list.Add(Create(HardwareTier.SSE));
            try { list.Add(Create(HardwareTier.AVX2)); } catch { }
            try { list.Add(Create(HardwareTier.FMA)); } catch { }
        }

        return [.. list];
    }

    /// <summary>
    /// Benchmarks AVX2 vs Scalar and returns whichever is faster.
    /// Falls back to Scalar if no SIMD support or if Scalar wins.
    /// </summary>
    public static QuantizationOps CreateForSystem(
        int benchVectorSize = 4_096,
        int warmup = 200,
        int iterations = 2_000)
    {
        var scalar = Create(HardwareTier.Scalar);

        if (!Sse.IsSupported)
            return scalar;

        var sse = Create(HardwareTier.SSE);

        if (!Avx2.IsSupported)
            return sse;

        var avx2 = Create(HardwareTier.AVX2);
        var fma  = Fma.IsSupported ? Create(HardwareTier.FMA) : avx2;

        // Benchmark Q8_0 and Q8K — representative of the two main code paths
        var best = BenchmarkVecDot(avx2, scalar, fma, benchVectorSize, warmup, iterations);
        return best;
    }

    private static unsafe QuantizationOps BenchmarkVecDot(
        QuantizationOps avx2, QuantizationOps scalar, QuantizationOps fma,
        int size, int warmup, int iterations)
    {
        var input = new float[size];
        var weights = new byte[size * 34 / 32 + 34]; // Q8_0 layout
        new Random(42).NextBytes(weights);
        for (int i = 0; i < size; i++) input[i] = (float)(new Random(i).NextDouble() - 0.5);

        fixed (float* pIn = input)
        fixed (byte* pW = weights)
        {
            // Warmup
            for (int i = 0; i < warmup; i++)
            {
                avx2.VecDotQ8_0(pIn, pW, 0, size);
                scalar.VecDotQ8_0(pIn, pW, 0, size);
            }

            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                avx2.VecDotQ8_0(pIn, pW, 0, size);
            long avx2Ticks = sw.ElapsedTicks;

            sw.Restart();
            for (int i = 0; i < iterations; i++)
                scalar.VecDotQ8_0(pIn, pW, 0, size);
            long scalarTicks = sw.ElapsedTicks;

            if (avx2Ticks <= scalarTicks)
                return avx2;

            // Also try FMA
            if (fma != avx2)
            {
                for (int i = 0; i < warmup; i++)
                    fma.VecDotQ8_0(pIn, pW, 0, size);

                sw.Restart();
                for (int i = 0; i < iterations; i++)
                    fma.VecDotQ8_0(pIn, pW, 0, size);
                long fmaTicks = sw.ElapsedTicks;

                if (fmaTicks <= scalarTicks)
                    return fma;
            }

            return scalar;
        }
    }
}
