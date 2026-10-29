using BenchmarkDotNet.Attributes;
using System.Numerics;

namespace SharpMind.Benchmarks;

/// <summary>
/// One head's DeltaNet state read (pred = stateᵀ·key) at Qwen3.5 head size:
/// the column-wise loop it replaced against the row-wise reduction now used.
/// </summary>
[MemoryDiagnoser]
public class DeltaNetRecurrenceBenchmarks
{
    [Params(64, 128)]
    public int HeadDim { get; set; }

    private float[] _state = [];
    private float[] _key = [];
    private float[] _acc = [];

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(3);
        _state = Enumerable.Range(0, HeadDim * HeadDim).Select(_ => (float)rng.NextDouble()).ToArray();
        _key = Enumerable.Range(0, HeadDim).Select(_ => (float)rng.NextDouble()).ToArray();
        _acc = new float[HeadDim];
    }

    // Mirrors SharpMind.Model DeltaNetLayer.MulAddRow (internal to that assembly).
    private static void MulAddRow(Span<float> dst, ReadOnlySpan<float> row, float s)
    {
        int i = 0;
        var sv = new Vector<float>(s);
        for (; i <= dst.Length - Vector<float>.Count; i += Vector<float>.Count)
            (new Vector<float>(dst.Slice(i)) + new Vector<float>(row.Slice(i)) * sv).CopyTo(dst.Slice(i));
        for (; i < dst.Length; i++) dst[i] += row[i] * s;
    }

    [Benchmark(Baseline = true)]
    public float ColumnWise()
    {
        for (int v = 0; v < HeadDim; v++)
        {
            float acc = 0f;
            for (int k = 0; k < HeadDim; k++)
                acc += _state[k * HeadDim + v] * _key[k];
            _acc[v] = acc;
        }
        return _acc[0];
    }

    [Benchmark]
    public float RowWise()
    {
        _acc.AsSpan().Clear();
        for (int k = 0; k < HeadDim; k++)
            MulAddRow(_acc, _state.AsSpan(k * HeadDim, HeadDim), _key[k]);
        return _acc[0];
    }
}
