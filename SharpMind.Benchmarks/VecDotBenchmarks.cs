using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using SharpMind.Core.Quantization;
using SharpMind.GPU;

namespace SharpMind.Benchmarks;

public static class VecDotHelper
{
    public static QuantizationOps CreateScalar() => QuantizationFactory.Create(HardwareTier.Scalar);
    public static QuantizationOps CreateSse()    => Sse3.IsSupported ? QuantizationFactory.Create(HardwareTier.SSE) : CreateScalar();
    public static QuantizationOps CreateAvx2()   => Avx2.IsSupported ? QuantizationFactory.Create(HardwareTier.AVX2) : CreateSse();
    public static QuantizationOps CreateFma()    => Fma.IsSupported  ? QuantizationFactory.Create(HardwareTier.FMA)  : CreateAvx2();

    public static Func<float> TryCreateGpu(Func<float> kernel)
    {
        try { var _ = kernel(); return kernel; }
        catch { return static () => float.NaN; }
    }
}

[SimpleJob(RuntimeMoniker.Net90, warmupCount: 5, iterationCount: 10)]
[MemoryDiagnoser]
public unsafe class Q8_0Benchmarks
{
    private const int Size = 8192;
    private QuantizationOps _scalar, _sse, _avx2, _fma;
    private Func<float> _gpu;
    private float* _input;
    private byte* _w;

    [GlobalSetup]
    public void Setup()
    {
        _scalar = VecDotHelper.CreateScalar(); _sse = VecDotHelper.CreateSse();
        _avx2 = VecDotHelper.CreateAvx2(); _fma = VecDotHelper.CreateFma();
        var rng = new Random(42);
        _input = (float*)NativeMemory.AlignedAlloc((nuint)(Size * sizeof(float)), 32);
        for (int i = 0; i < Size; i++) _input[i] = (float)(rng.NextDouble() - 0.5);
        int q80bytes = (Size / 32) * 34 + 34;
        _w = (byte*)NativeMemory.Alloc((nuint)q80bytes);
        new Random(42).NextBytes(new Span<byte>(_w, q80bytes));
        ushort d = BitConverter.HalfToUInt16Bits((Half)0.1f);
        for (int b = 0; b < Size / 32; b++)
        { _w[b * 34 + 0] = (byte)(d & 0xFF); _w[b * 34 + 1] = (byte)(d >> 8); }
        _gpu = VecDotHelper.TryCreateGpu(() => GPUQuantizationKernels.VecDotQ8_0_GPU(_input, _w, 0, Size));
    }

    [GlobalCleanup]
    public void Cleanup() { NativeMemory.Free(_w); NativeMemory.AlignedFree(_input); }

    [Benchmark(Baseline = true)] public float Scalar() => _scalar.VecDotQ8_0(_input, _w, 0, Size);
    [Benchmark] public float SSE()   => _sse.VecDotQ8_0(_input, _w, 0, Size);
    [Benchmark] public float AVX2()  => _avx2.VecDotQ8_0(_input, _w, 0, Size);
    [Benchmark] public float FMA()   => _fma.VecDotQ8_0(_input, _w, 0, Size);
    [Benchmark] public float GPU()   => _gpu();
}

[SimpleJob(RuntimeMoniker.Net90, warmupCount: 5, iterationCount: 10)]
[MemoryDiagnoser]
public unsafe class Q3KBenchmarks
{
    private const int Size = 8192;
    private QuantizationOps _scalar, _sse, _avx2, _fma;
    private Func<float> _gpu;
    private float* _input;
    private byte* _w;

    [GlobalSetup]
    public void Setup()
    {
        _scalar = VecDotHelper.CreateScalar(); _sse = VecDotHelper.CreateSse();
        _avx2 = VecDotHelper.CreateAvx2(); _fma = VecDotHelper.CreateFma();
        int bytes = (Size / 256) * 110 + 110;
        var rng = new Random(42);
        _input = (float*)NativeMemory.AlignedAlloc((nuint)(Size * sizeof(float)), 32);
        for (int i = 0; i < Size; i++) _input[i] = (float)(rng.NextDouble() - 0.5);
        _w = (byte*)NativeMemory.Alloc((nuint)bytes);
        new Random(42).NextBytes(new Span<byte>(_w, bytes));
        _gpu = VecDotHelper.TryCreateGpu(() => GPUQuantizationKernels.VecDotQ3K_GPU(_input, _w, 0, Size));
    }

    [GlobalCleanup]
    public void Cleanup() { NativeMemory.Free(_w); NativeMemory.AlignedFree(_input); }

    [Benchmark(Baseline = true)] public float Scalar() => _scalar.VecDotQ3K(_input, _w, 0, Size);
    [Benchmark] public float SSE()   => _sse.VecDotQ3K(_input, _w, 0, Size);
    [Benchmark] public float AVX2()  => _avx2.VecDotQ3K(_input, _w, 0, Size);
    [Benchmark] public float FMA()   => _fma.VecDotQ3K(_input, _w, 0, Size);
    [Benchmark] public float GPU()   => _gpu();
}

[SimpleJob(RuntimeMoniker.Net90, warmupCount: 5, iterationCount: 10)]
[MemoryDiagnoser]
public unsafe class Q4KBenchmarks
{
    private const int Size = 8192;
    private QuantizationOps _scalar, _sse, _avx2, _fma;
    private Func<float> _gpu;
    private float* _input;
    private byte* _w;

    [GlobalSetup]
    public void Setup()
    {
        _scalar = VecDotHelper.CreateScalar(); _sse = VecDotHelper.CreateSse();
        _avx2 = VecDotHelper.CreateAvx2(); _fma = VecDotHelper.CreateFma();
        int bytes = (Size / 256) * 144 + 144;
        var rng = new Random(42);
        _input = (float*)NativeMemory.AlignedAlloc((nuint)(Size * sizeof(float)), 32);
        for (int i = 0; i < Size; i++) _input[i] = (float)(rng.NextDouble() - 0.5);
        _w = (byte*)NativeMemory.Alloc((nuint)bytes);
        new Random(42).NextBytes(new Span<byte>(_w, bytes));
        _gpu = VecDotHelper.TryCreateGpu(() => GPUQuantizationKernels.VecDotQ4K_GPU(_input, _w, 0, Size));
    }

    [GlobalCleanup]
    public void Cleanup() { NativeMemory.Free(_w); NativeMemory.AlignedFree(_input); }

    [Benchmark(Baseline = true)] public float Scalar() => _scalar.VecDotQ4K(_input, _w, 0, Size);
    [Benchmark] public float SSE()   => _sse.VecDotQ4K(_input, _w, 0, Size);
    [Benchmark] public float AVX2()  => _avx2.VecDotQ4K(_input, _w, 0, Size);
    [Benchmark] public float FMA()   => _fma.VecDotQ4K(_input, _w, 0, Size);
    [Benchmark] public float GPU()   => _gpu();
}
