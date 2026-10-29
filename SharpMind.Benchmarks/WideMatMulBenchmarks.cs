using BenchmarkDotNet.Attributes;
using SharpMind.Core.Quantization;

namespace SharpMind.Benchmarks;

/// <summary>
/// Q8_0 decode matmul (m = 1) at the Qwen2-0.5B shapes: the wide parallel kernel,
/// the same kernel forced onto one chunk, and the serial default kernel.
/// Reports bytes-per-second of raw Q8_0 weights read, so results compare with
/// the machine's memory bandwidth. On a laptop, run-to-run variance is large;
/// compare variants within one run, interleaved, not across runs.
/// </summary>
[MemoryDiagnoser]
public unsafe class WideMatMulBenchmarks
{
    /// <summary>0 = FFN gate/up (K=896, N=4864); 1 = FFN down (K=4864, N=896).</summary>
    [Params(0, 1)]
    public int Shape { get; set; }

    private int _k, _n;
    private byte[] _raw = [];
    private Q8_0WideWeights? _wide;
    private float[] _input = [], _output = [];

    [GlobalSetup]
    public void Setup()
    {
        (_k, _n) = Shape == 0 ? (896, 4864) : (4864, 896);
        int blocks = _k / 32;
        _raw = new byte[(long)_n * blocks * 34];
        new Random(1).NextBytes(_raw);
        ushort scale = BitConverter.HalfToUInt16Bits((Half)0.01f);
        for (long i = 0; i < (long)_n * blocks; i++)
        {
            _raw[i * 34] = (byte)scale;
            _raw[i * 34 + 1] = (byte)(scale >> 8);
        }
        _wide = Q8_0WideWeights.TryCreate(_raw, _k, _n);
        _input = new float[_k];
        for (int i = 0; i < _k; i++) _input[i] = (float)Math.Sin(i);
        _output = new float[_n];
        _wide?.MatMul(Ptr(_input), Ptr(_output), 1); // build the repack outside the timed region
    }

    private static float* Ptr(float[] a) => (float*)System.Runtime.CompilerServices.Unsafe.AsPointer(ref a[0]);

    [Benchmark(Baseline = true)]
    public float WideParallel()
    {
        Q8_0WideWeights.ParallelChunkOverride = 0;
        _wide!.MatMul(Ptr(_input), Ptr(_output), 1);
        return _output[0];
    }

    [Benchmark]
    public float WideSingleChunk()
    {
        Q8_0WideWeights.ParallelChunkOverride = 1;
        try { _wide!.MatMul(Ptr(_input), Ptr(_output), 1); }
        finally { Q8_0WideWeights.ParallelChunkOverride = 0; }
        return _output[0];
    }

    [Benchmark]
    public float SerialDefaultKernel()
    {
        fixed (byte* pRaw = _raw)
            QuantizationKernels.QuantizedMatMulQ8_0_Serial_FMA(Ptr(_input), pRaw, Ptr(_output), 1, _k, _n);
        return _output[0];
    }
}
