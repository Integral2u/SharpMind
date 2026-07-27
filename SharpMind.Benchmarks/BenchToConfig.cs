using SharpMind.Core;
using SharpMind.Core.Quantization;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;

namespace SharpMind.Benchmarks;

/// <summary>
/// Opt-in tool: benchmarks each key quantization op across FMA/AVX2/SSE/Scalar
/// and outputs a per-op JSON config. Run via:
///   dotnet run --project SharpMind.Benchmarks -- --bench-to-config [output.json]
/// The output feeds into QuantizationFactory.Create(deserialize the JSON).
/// </summary>
internal static class BenchToConfig
{
    private const int VecSize = 4096;
    private const int Warmup = 500;
    private const int Iterations = 5000;

    public static void Run(string outputPath)
    {
        // Collect available tiers
        var tiers = new List<(string Suffix, QuantizationOps Ops)>();
        void AddIf(HardwareTier hw, string suffix, bool supported)
        {
            if (supported) tiers.Add((suffix, QuantizationFactory.Create(hw)));
        }

        AddIf(HardwareTier.FMA, "_fma", Fma.IsSupported);
        AddIf(HardwareTier.AVX2, "_avx2", Avx2.IsSupported);
        AddIf(HardwareTier.SSE, "_sse", Sse.IsSupported);
        tiers.Add(("_scalar", QuantizationFactory.Create(HardwareTier.Scalar)));

        var mapping = new Dictionary<string, string>();

        // The suffix for ops where all SIMD tiers map to the same impl.
        string simdSuffix = Fma.IsSupported ? "_fma"
                          : Avx2.IsSupported ? "_avx2"
                          : Sse.IsSupported  ? "_sse"
                          :                    "_scalar";

        Console.WriteLine($"Benchmarking {tiers.Count} tiers across {VecSize} elements...");

        // ── VecDot ops (benchmarked per-op) ──────────────────────────────
        var rng = new Random(42);
        unsafe
        {
            float* input = AllocInput(VecSize, rng);
            byte* weights = AllocWeights(VecSize * 2, rng);

            var vecDotEntries = new (string Key, string Prefix, Action<QuantizationOps> BenchmarkAction)[]
            {
                (QuantizationKeys.KeyVecDotQ8_0, "q8_0", ops => { ops.VecDotQ8_0(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ8K,  "q8k",  ops => { ops.VecDotQ8K(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ4K,  "q4k",  ops => { ops.VecDotQ4K(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ6K,  "q6k",  ops => { ops.VecDotQ6K(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ3K,  "q3k",  ops => { ops.VecDotQ3K(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ2K,  "q2k",  ops => { ops.VecDotQ2K(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ5K,  "q5k",  ops => { ops.VecDotQ5K(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ4_0, "q4_0", ops => { ops.VecDotQ4_0(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ4_1, "q4_1", ops => { ops.VecDotQ4_1(input, weights, 0, VecSize); }),
                (QuantizationKeys.KeyVecDotQ8_1, "q8_1", ops => { ops.VecDotQ8_1(input, weights, 0, VecSize); }),
            };

            foreach (var (key, prefix, benchmark) in vecDotEntries)
            {
                var (suffix, _) = BenchmarkBestTier(tiers, benchmark);
                mapping[key] = $"{prefix}{suffix}";
                Console.WriteLine($"  {key,-22} → {mapping[key]}");
            }

            Marshal.FreeHGlobal((IntPtr)input);
            Marshal.FreeHGlobal((IntPtr)weights);
        }

        // ── QuantizedMatMulQ8_0 (different signature) ───────────────────
        unsafe
        {
            int k = VecSize, n = 256;
            int nBlocks = (k + 31) / 32;
            const int BlockBytes = 34;
            int colStride = nBlocks * BlockBytes;
            int totalWeightBytes = n * colStride;
            float* input = AllocInput(k, rng);
            byte* w = (byte*)Marshal.AllocHGlobal(totalWeightBytes);
            new Random(42).NextBytes(new Span<byte>(w, totalWeightBytes));
            // Set valid half scale (0.1f) for each Q8_0 block across all columns
            ushort scale = BitConverter.HalfToUInt16Bits((Half)0.1f);
            for (int col = 0; col < n; col++)
                for (int b = 0; b < nBlocks; b++)
                { w[col * colStride + b * BlockBytes + 0] = (byte)(scale & 0xFF); w[col * colStride + b * BlockBytes + 1] = (byte)(scale >> 8); }
            float* output = (float*)Marshal.AllocHGlobal(n * sizeof(float));

            Action<QuantizationOps> matMulAction = ops =>
            {
                ops.QuantizedMatMulQ8_0(input, w, output, 1, k, n);
            };

            var (matMulSuffix, _) = BenchmarkBestTier(tiers, matMulAction);
            mapping[QuantizationKeys.KeyQuantizedMatMulQ8_0] = $"qmatmul_q8_0{matMulSuffix}";
            Console.WriteLine($"  {QuantizationKeys.KeyQuantizedMatMulQ8_0,-22} → {mapping[QuantizationKeys.KeyQuantizedMatMulQ8_0]}");

            Marshal.FreeHGlobal((IntPtr)input);
            Marshal.FreeHGlobal((IntPtr)w);
            Marshal.FreeHGlobal((IntPtr)output);
        }

        // ── Shared helpers (inferred from best SIMD tier) ────────────────
        mapping[QuantizationKeys.KeyHSum256]     = $"hsum{simdSuffix}";
        mapping[QuantizationKeys.KeyHalfToFloat] = $"halftofloat{simdSuffix}";
        mapping[QuantizationKeys.KeyFloatToHalf] = $"floattohalf{simdSuffix}";

        // Always scalar (no SIMD variant exists)
        mapping[QuantizationKeys.KeyVecDotQ5_0] = "q5_0_scalar";
        mapping[QuantizationKeys.KeyVecDotQ5_1] = "q5_1_scalar";
        mapping[QuantizationKeys.KeyGetScaleMinK4_Scale] = "getscalemink4_scale_scalar";
        mapping[QuantizationKeys.KeyGetScaleMinK4_Min]   = "getscalemink4_min_scalar";

        // ── Write JSON ───────────────────────────────────────────────────
        var json = JsonSerializer.Serialize(mapping, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(outputPath, json);
        Console.WriteLine($"\nConfig written to {outputPath}");
    }

    private static (string Suffix, QuantizationOps Ops) BenchmarkBestTier(
        List<(string Suffix, QuantizationOps Ops)> tiers,
        Action<QuantizationOps> benchmark)
    {
        // Warmup all tiers
        for (int i = 0; i < Warmup; i++)
            foreach (var (_, ops) in tiers)
                benchmark(ops);

        long bestTime = long.MaxValue;
        (string Suffix, QuantizationOps Ops) best = default;

        foreach (var (suffix, ops) in tiers)
        {
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < Iterations; i++)
                benchmark(ops);
            sw.Stop();

            long elapsed = sw.ElapsedTicks;
            Console.WriteLine($"    {suffix,-10} {elapsed,10} ticks");

            if (elapsed < bestTime)
            {
                bestTime = elapsed;
                best = (suffix, ops);
            }
        }

        return best;
    }

    private static unsafe float* AllocInput(int size, Random rng)
    {
        var ptr = (float*)Marshal.AllocHGlobal(size * sizeof(float));
        for (int i = 0; i < size; i++)
            ptr[i] = (float)(rng.NextDouble() - 0.5);
        return ptr;
    }

    private static unsafe byte* AllocWeights(int size, Random rng)
    {
        var ptr = (byte*)Marshal.AllocHGlobal(size);
        var span = new Span<byte>(ptr, size);
        rng.NextBytes(span);
        return ptr;
    }
}
