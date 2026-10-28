using BenchmarkDotNet.Attributes;
using SharpMind.Inference;

namespace SharpMind.Benchmarks;

/// <summary>
/// Per-token sampling cost over realistic vocabulary sizes. Reports time and
/// allocations per Sample call; allocations should be zero in steady state.
/// </summary>
[MemoryDiagnoser]
public class SamplerBenchmarks
{
    [Params(32_000, 151_936)]
    public int VocabSize { get; set; }

    private float[] _logits = [];
    private Random _rng = null!;
    private SamplingConfig _unfiltered = null!;
    private SamplingConfig _topKTopP = null!;

    [GlobalSetup]
    public void Setup()
    {
        var data = new Random(42);
        _logits = new float[VocabSize];
        for (int i = 0; i < _logits.Length; i++) _logits[i] = (float)(data.NextDouble() * 20 - 10);
        _rng = new Random(7);
        _unfiltered = new SamplingConfig { Temperature = 0.8f, TopK = 0, TopP = 1f };
        _topKTopP = new SamplingConfig { Temperature = 0.8f, TopK = 40, TopP = 0.9f };
    }

    [Benchmark(Baseline = true)]
    public int Greedy()
    {
        return Sampler.Sample(_logits, SamplingConfig.Greedy, _rng);
    }

    [Benchmark]
    public int Unfiltered()
    {
        return Sampler.Sample(_logits, _unfiltered, _rng);
    }

    [Benchmark]
    public int TopK40TopP090()
    {
        return Sampler.Sample(_logits, _topKTopP, _rng);
    }
}
