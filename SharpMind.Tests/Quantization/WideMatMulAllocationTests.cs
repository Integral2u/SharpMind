using SharpMind.Core.Quantization;
using Xunit;
using Xunit.Abstractions;

namespace SharpMind.Tests.Quantization;

/// <summary>
/// Measures the process-wide managed allocation per wide Q8_0 matmul at the
/// Qwen2-0.5B FFN decode shape. Diagnostic: it records the figure and asserts
/// only that it is finite, so a regression shows up in the test output.
/// </summary>
public class WideMatMulAllocationTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(896, 4864)]   // FFN gate/up projection
    [InlineData(4864, 896)]   // FFN down projection
    public unsafe void SerialDefaultKernel_AllocationPerCall(int k, int n)
    {
        // The default decode kernel for CPU (compound q8_0_serial_fma), M = 1.
        int blocks = k / 32;
        var raw = new byte[(long)n * blocks * 34];
        new Random(2).NextBytes(raw);
        for (long i = 0; i < (long)n * blocks; i++)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)0.01f);
            raw[i * 34] = (byte)bits;
            raw[i * 34 + 1] = (byte)(bits >> 8);
        }
        var input = new float[k];
        var result = new float[n];
        fixed (float* pIn = input)
        fixed (float* pOut = result)
        fixed (byte* pW = raw)
        {
            for (int i = 0; i < 50; i++) QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA(pIn, pW, pOut, 1, k, n);
            const int iters = 2000;
            long total = GC.GetTotalAllocatedBytes(precise: true);
            for (int i = 0; i < iters; i++) QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA(pIn, pW, pOut, 1, k, n);
            long totalAfter = GC.GetTotalAllocatedBytes(precise: true);
            output.WriteLine($"SERIAL K={k} N={n}: {(totalAfter - total) / (double)iters:F0} B/call");
            for (int i = 0; i < 50; i++) QuantizationKernels.QuantizedMatMulQ8_0_Parallel_FMA(pIn, pW, pOut, 1, k, n);
            long t2 = GC.GetTotalAllocatedBytes(precise: true);
            for (int i = 0; i < iters; i++) QuantizationKernels.QuantizedMatMulQ8_0_Parallel_FMA(pIn, pW, pOut, 1, k, n);
            output.WriteLine($"PARALLEL K={k} N={n}: {(GC.GetTotalAllocatedBytes(precise: true) - t2) / (double)iters:F0} B/call");
        }
    }

    [Theory]
    [InlineData(896, 4864)]   // FFN gate/up projection
    [InlineData(4864, 896)]   // FFN down projection
    public unsafe void DecodeShape_AllocationPerCall(int k, int n)
    {
        if (!Q8_0WideWeights.IsSupported(k, n)) return;

        var rng = new Random(1);
        int blocks = k / 32;
        var raw = new byte[(long)n * blocks * 34];
        rng.NextBytes(raw);
        // Keep the fp16 scales finite: set each block's scale to 0.01.
        for (long i = 0; i < (long)n * blocks; i++)
        {
            ushort bits = (ushort)BitConverter.HalfToUInt16Bits((Half)0.01f);
            raw[i * 34] = (byte)bits;
            raw[i * 34 + 1] = (byte)(bits >> 8);
        }
        var wide = Q8_0WideWeights.TryCreate(raw, k, n)!;
        var input = new float[k];
        for (int i = 0; i < k; i++) input[i] = (float)(rng.NextDouble() * 2 - 1);
        var result = new float[n];

        fixed (float* pIn = input)
        fixed (float* pOut = result)
        {
            for (int i = 0; i < 50; i++) wide.MatMul(pIn, pOut, 1); // warm up: JIT, pools

            const int iters = 2000;
            long before = GC.GetAllocatedBytesForCurrentThread();
            long total = GC.GetTotalAllocatedBytes(precise: true);
            for (int i = 0; i < iters; i++) wide.MatMul(pIn, pOut, 1);
            long totalAfter = GC.GetTotalAllocatedBytes(precise: true);

            double perCall = (totalAfter - total) / (double)iters;
            output.WriteLine($"K={k} N={n}: {perCall:F0} B/call (process-wide)");
            Assert.True(perCall >= 0);
        }
    }
}
