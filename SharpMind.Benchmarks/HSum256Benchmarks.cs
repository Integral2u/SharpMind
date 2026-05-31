using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;
using SharpMind.Core;

namespace SharpMind.Benchmarks;

[SimpleJob(RuntimeMoniker.Net90, warmupCount: 5, iterationCount: 10)]
[MemoryDiagnoser]
public class HSum256Benchmarks
{
    private Vector256<float> _v;

    [GlobalSetup]
    public void Setup()
    {
        _v = Vector256.Create(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f);
    }

    [Benchmark(Baseline = true)]
    public float Scalar() => MathHelpers.HSum256_Scalar(_v);

    [Benchmark]
    public float HSumAvx() => Avx.IsSupported ? MathHelpers.HSum256_Avx(_v) : Scalar();

    [Benchmark]
    public float HSumSse3() => Sse3.IsSupported ? MathHelpers.HSum256_Sse3(_v) : Scalar();
}
